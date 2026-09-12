"""Second, independent pre-annotation pass over the same golden items.

Differences from pass A, on purpose:
  * does NOT see the brand's historical reply (pass A did) -- so the two passes
    cannot agree merely by both copying the brand's behaviour,
  * asks the decision question first and the taxonomy question second,
  * different model family role: A used the judge model, B uses the generator
    model, so agreement is not one model agreeing with itself.

Where A and B disagree, the item is forced into the human adjudication queue.
That is the whole point: the disagreement rate tells us which items are
genuinely ambiguous rather than relying on one model's self-reported confidence.
"""
import concurrent.futures as cf, json, pathlib, sys, collections

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm import LLM, extract_json, GEN_MODEL
from taxonomy import INTENTS, ESCALATION_RULES

SYS = ("You are a support-operations reviewer deciding how incoming public tweets "
       "should be routed. Output JSON only.")

TMPL = """A customer sent this message to Amazon's public support account on Twitter.

FIRST decide: can a bot safely answer this in public without a human, or must a
human take it over? A human is required if ANY of these applies:
{rules}

THEN pick the single best category:
{intents}

MESSAGE: {msg}
{ctx}
Return JSON:
{{"action": "auto" | "escalate",
  "rules_fired": ["<rule key>", ...],
  "intent": "<one category key>",
  "reason": "<one sentence>"}}"""


def main() -> None:
    llm = LLM()
    intents = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    rules = "\n".join(f"- {k}: {v}" for k, v in ESCALATION_RULES.items())
    rows = [json.loads(l) for l in open("data/golden/preannotated.jsonl")]
    # Successful calls are cached, so a rerun only redoes what actually failed.

    def annotate(r):
        ctx = f"EARLIER IN THREAD: {r['prior_turn']}\n" if r.get("prior_turn") else ""
        p = TMPL.format(rules=rules, intents=intents, msg=r["customer_msg"], ctx=ctx)
        try:
            d = extract_json(llm.complete(p, system=SYS, model=GEN_MODEL, max_tokens=500))
            if d.get("intent") in INTENTS and d.get("action") in ("auto", "escalate"):
                return {**r, "pre_b": d}
        except Exception as e:
            print(f"  pass B failed on {r['case_id']}: {str(e)[:100]}", file=sys.stderr)
        return {**r, "pre_b": {"intent": None, "action": None, "rules_fired": [],
                               "reason": "PASS B FAILED"}}

    with cf.ThreadPoolExecutor(5) as ex:
        out = list(ex.map(annotate, rows))

    with open("data/golden/preannotated.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    ai = [r["pre"]["intent"] for r in out]
    bi = [r["pre_b"]["intent"] for r in out]
    aa = [r["pre"]["action"] for r in out]
    ba = [r["pre_b"]["action"] for r in out]
    agree_i = sum(x == y for x, y in zip(ai, bi))
    agree_a = sum(x == y for x, y in zip(aa, ba))
    n = len(out)
    print(f"n={n}", file=sys.stderr)
    print(f"intent agreement A/B: {agree_i}/{n} = {agree_i/n:.1%}", file=sys.stderr)
    print(f"action agreement A/B: {agree_a}/{n} = {agree_a/n:.1%}", file=sys.stderr)
    conflicts = [(r["pre"]["intent"], r["pre_b"]["intent"]) for r in out
                 if r["pre"]["intent"] != r["pre_b"]["intent"]]
    print("top intent conflicts:", collections.Counter(conflicts).most_common(12), file=sys.stderr)
    print("llm:", llm.stats(), file=sys.stderr)


if __name__ == "__main__":
    main()
