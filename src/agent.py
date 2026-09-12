"""The support agent: classify -> retrieve -> draft -> route.

Design choices that the evaluation then has to justify:
  * One LLM call does intent + escalation + reply, because they are not
    independent decisions: what you can safely say depends on whether a human
    must act. Two calls let them contradict each other.
  * Retrieval happens BEFORE the call, and the model is told to ground the reply
    in the retrieved replies. Empty/weak retrieval is itself an escalation
    trigger (E6), so a confident ungrounded answer is not reachable.
  * Deterministic post-checks override the model. A model that says "auto" on a
    refund cannot auto-post, regardless of how confident it sounds.
"""
from __future__ import annotations
import json, re, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from llm import LLM, extract_json, GEN_MODEL
from retrieval import CaseRetriever
from taxonomy import (INTENTS, ESCALATION_RULES, ALWAYS_ESCALATE, default_action)

# Below this top-1 similarity there is no usable precedent -> E6.
GROUNDING_FLOOR = 0.13

SYSTEM = (
    "You are the AI triage layer for Amazon's customer-support Twitter account. "
    "You classify the incoming message, decide whether it can be answered without "
    "a human, and draft the public reply. You never invent order details, refund "
    "amounts, dates, or policies. Output JSON only."
)

PROMPT = """INTENTS (choose exactly one key):
{intents}

ESCALATION RULES (if ANY applies, action must be "escalate"):
{rules}

HOW THIS BRAND HAS REPLIED TO SIMILAR MESSAGES (your only source of truth for
tone and for what the brand actually offers; similarity score in brackets):
{examples}

INCOMING MESSAGE: {msg}
{ctx}
Write the reply the way this brand writes: brief enough for one tweet, name the
concrete next step, no invented specifics. If a human must take over, the reply
should acknowledge and set expectations WITHOUT promising an outcome.

Return JSON:
{{"intent": "<one key>",
  "confidence": <0.0-1.0>,
  "action": "auto" | "escalate",
  "rules_fired": ["<rule key>", ...],
  "reason": "<one sentence justifying the action>",
  "reply": "<the reply text, max 280 characters>"}}"""


class SupportAgent:
    def __init__(self, retriever: CaseRetriever | None = None, llm: LLM | None = None,
                 k: int = 4, confidence_floor: float = 0.55):
        self.r = retriever if retriever is not None else CaseRetriever()
        self.llm = llm or LLM()
        self.k = k
        self.confidence_floor = confidence_floor
        self._intents = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
        self._rules = "\n".join(f"- {k}: {v}" for k, v in ESCALATION_RULES.items())

    def handle(self, msg: str, prior_turn: str = "") -> dict:
        hits = self.r.search(msg, k=self.k)
        examples = "\n\n".join(
            f"[{h['score']:.2f}] customer: {h['customer_msg']}\n      brand: {h['brand_reply']}"
            for h in hits) or "(no similar case found)"
        p = PROMPT.format(intents=self._intents, rules=self._rules, examples=examples,
                          msg=msg, ctx=f"EARLIER IN THREAD: {prior_turn}\n" if prior_turn else "")
        raw = self.llm.complete(p, system=SYSTEM, model=GEN_MODEL, max_tokens=800)
        try:
            d = extract_json(raw)
        except ValueError:
            d = {}
        return self._enforce(d, hits, msg)

    # -- deterministic guardrails ------------------------------------------
    def _enforce(self, d: dict, hits: list[dict], msg: str) -> dict:
        intent = d.get("intent") if d.get("intent") in INTENTS else None
        try:
            conf = min(1.0, max(0.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        action = d.get("action") if d.get("action") in ("auto", "escalate") else None
        fired = [r for r in (d.get("rules_fired") or []) if r in ESCALATION_RULES]
        reply = (d.get("reply") or "").strip()
        reasons = []

        if intent is None:                       # unparseable -> cannot be trusted
            intent, conf, action = "no_request", 0.0, "escalate"
            fired.append("E7_low_confidence")
            reasons.append("classifier output unusable")

        top = hits[0]["score"] if hits else 0.0
        if top < GROUNDING_FLOOR:
            action = "escalate"
            if "E6_no_grounding" not in fired:
                fired.append("E6_no_grounding")
            reasons.append(f"top similarity {top:.2f} below grounding floor {GROUNDING_FLOOR}")
        if conf < self.confidence_floor:
            action = "escalate"
            if "E7_low_confidence" not in fired:
                fired.append("E7_low_confidence")
            reasons.append(f"confidence {conf:.2f} below floor {self.confidence_floor}")
        if intent in ALWAYS_ESCALATE:
            action = "escalate"
            if not fired:
                fired.append("E1_account_or_pii")
            reasons.append(f"intent '{intent}' is never auto-handled by policy")
        if fired and action != "escalate":        # model contradicted itself
            action = "escalate"
            reasons.append("escalation rule fired but action said auto")
        if action is None:
            action = default_action(intent)

        if len(reply) > 280:
            reply = reply[:277].rstrip() + "..."
        # A reply that invents a specific promise is not safe to auto-post.
        if action == "auto" and re.search(
                r"\b(refund(ed)?|credit(ed)?|compensat|replac(e|ed|ement)|"
                r"deliver(ed)?\s+(today|tomorrow|by)|guarantee)\b", reply, re.I):
            action = "escalate"
            fired.append("E2_money_commitment")
            reasons.append("draft reply commits money/goods, needs a human")

        return dict(intent=intent, confidence=conf, action=action,
                    rules_fired=sorted(set(fired)),
                    reason=(d.get("reason") or "").strip(),
                    guardrail_reasons=reasons, reply=reply,
                    top_similarity=round(top, 4),
                    retrieved=[{"case_id": h["case_id"], "score": h["score"]} for h in hits])


def demo():
    """Guardrails must hold without touching the network."""
    class FakeR:
        def search(self, q, k=4):
            return [dict(case_id="x", customer_msg="q", brand_reply="a", score=0.9)]
    class FakeLLM:
        def __init__(self, payload): self.payload = payload
        def complete(self, *a, **k): return json.dumps(self.payload)

    def run(payload, **kw):
        return SupportAgent(FakeR(), FakeLLM(payload), **kw).handle("some message")

    # policy overrides a confident model
    r = run(dict(intent="refund_status", confidence=0.99, action="auto",
                 rules_fired=[], reason="", reply="All set!"))
    assert r["action"] == "escalate" and "never auto-handled" in " ".join(r["guardrail_reasons"])
    # low confidence escalates
    r = run(dict(intent="product_info", confidence=0.2, action="auto", reply="Yes it ships."))
    assert r["action"] == "escalate" and "E7_low_confidence" in r["rules_fired"]
    # unparseable output escalates instead of crashing
    r = SupportAgent(FakeR(), type("L", (), {"complete": lambda *a, **k: "not json"})()).handle("m")
    assert r["action"] == "escalate" and r["confidence"] == 0.0
    # a money promise cannot be auto-posted
    r = run(dict(intent="delivery_late", confidence=0.95, action="auto",
                 rules_fired=[], reason="", reply="We will refund you in full today."))
    assert r["action"] == "escalate" and "E2_money_commitment" in r["rules_fired"]
    # a clean auto case stays auto
    r = run(dict(intent="product_info", confidence=0.9, action="auto",
                 rules_fired=[], reason="public info", reply="Yes, that model works in India."))
    assert r["action"] == "auto" and r["rules_fired"] == []
    # replies are truncated to one tweet
    r = run(dict(intent="product_info", confidence=0.9, action="auto", reply="x" * 400))
    assert len(r["reply"]) == 280
    print("agent guardrails ok")


if __name__ == "__main__":
    demo()
