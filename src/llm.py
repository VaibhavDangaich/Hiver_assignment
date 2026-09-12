"""Cached LLM client with two interchangeable backends.

Cache is keyed on (backend-independent) model + temperature + system + prompt, so:
  * interrupted runs resume for free,
  * the committed cache IS the reproduction artifact -- `make reproduce` replays it
    with no network and no API key.

Backends
  api : official Anthropic SDK, needs ANTHROPIC_API_KEY.  Use this if you have a key.
  cli : shells out to the local `claude -p` binary (Claude Code subscription auth).
        This is what generated the committed artifacts -- see README "Provenance".
"""
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, threading, time

CACHE_PATH = pathlib.Path(os.environ.get("LLM_CACHE", "artifacts/cache/llm_cache.jsonl"))
GEN_MODEL   = "claude-haiku-4-5-20251001"   # drafts + classifies
JUDGE_MODEL = "claude-sonnet-5"             # judges -- deliberately NOT the generator


def _key(model: str, temperature: float, system: str, prompt: str) -> str:
    h = hashlib.sha256()
    for part in (model, f"{temperature:.3f}", system, prompt):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


class LLM:
    def __init__(self, backend: str | None = None, cache_path: pathlib.Path | None = None,
                 offline: bool = False):
        self.backend = backend or os.environ.get("LLM_BACKEND", "cli")
        self.offline = offline or os.environ.get("LLM_OFFLINE") == "1"
        self.path = cache_path or CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.cache: dict[str, str] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.open():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                     # tolerate a torn final line
                self.cache[rec["key"]] = rec["response"]
        self._client = None

    # ---- backends -----------------------------------------------------------
    def _call_api(self, model, temperature, system, prompt, max_tokens):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        r = self._client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")

    def _call_cli(self, model, temperature, system, prompt, max_tokens):
        # `claude -p` has no temperature flag; prompts are written to be
        # deterministic-ish and the cache pins whatever came back.
        # Prompt goes on stdin: as argv it hits shell/arg-length limits and the
        # CLI silently mis-parses multi-line prompts.
        cmd = ["claude", "-p", "--model", model]
        if system:
            cmd += ["--append-system-prompt", system]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"claude cli failed: {r.stderr[-400:]}")
        return r.stdout.strip()

    # ---- public -------------------------------------------------------------
    def complete(self, prompt: str, system: str = "", model: str = GEN_MODEL,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        k = _key(model, temperature, system, prompt)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        if self.offline:
            raise KeyError(
                f"cache miss in offline mode (key {k}). The committed cache only "
                f"covers the pinned eval set; run without LLM_OFFLINE=1 to generate.")
        fn = self._call_api if self.backend == "api" else self._call_cli
        last = None
        for attempt in range(4):
            try:
                out = fn(model, temperature, system, prompt, max_tokens)
                break
            except Exception as e:                # transient 429/529/timeout
                last = e
                # Don't silently burn retries on a deterministic failure: a bad
                # prompt fails identically 4 times and just looks like slowness.
                if isinstance(e, (ValueError, TypeError)):
                    raise
                print(f"  [llm retry {attempt + 1}/4] {str(e)[:160]}", file=sys.stderr)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"LLM failed after 4 attempts: {last}")
        with self._lock:
            self.cache[k] = out
            self.misses += 1
            with self.path.open("a") as f:
                f.write(json.dumps(dict(key=k, model=model, temperature=temperature,
                                        backend=self.backend, system=system,
                                        prompt=prompt, response=out)) + "\n")
        return out

    def stats(self) -> dict:
        return dict(hits=self.hits, misses=self.misses, cached=len(self.cache))


def extract_json(text: str) -> dict:
    """Models wrap JSON in prose or fences more often than anyone admits."""
    t = text.strip()
    if "```" in t:
        seg = t.split("```")[1]
        t = seg[4:] if seg.lstrip().lower().startswith("json") else seg
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth, instr, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(t[start:i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")


def demo():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('here you go:\n```json\n{"a": [1,2]}\n```\nhope that helps') == {"a": [1, 2]}
    assert extract_json('{"t": "brace } inside string"}')["t"] == "brace } inside string"
    assert extract_json('{"t": "escaped \\" quote"}')["t"] == 'escaped " quote'
    assert extract_json('prose {"a": {"b": 2}} tail') == {"a": {"b": 2}}
    k1 = _key("m", 0.0, "s", "p"); k2 = _key("m", 0.0, "s", "p2")
    assert k1 != k2 and k1 == _key("m", 0.0, "s", "p")
    # key must not be confusable by concatenation (null-separated fields)
    assert _key("m", 0.0, "ab", "c") != _key("m", 0.0, "a", "bc")
    print("llm ok")


if __name__ == "__main__":
    demo()
