"""Baselines. The point of each is to be beatable in a *specific* way, so that
the agent's headline number has to explain what it is actually buying.

B0 majority   - trivial. Predicts the most frequent intent and always escalates.
                Exposes how much of the accuracy number is just class imbalance,
                and shows that "escalate everything" is a legitimate strategy
                under our 10:1 cost model -- it never makes the expensive error.
B1 keyword    - simple, no LLM. Hand-written rules over the same taxonomy.
                This is what a team would ship in an afternoon; the LLM has to
                beat it to justify its cost and latency.
B2 retrieval  - no-LLM reply generation: copy the reply the brand actually sent
                to the nearest historical case. Tests whether *generation* adds
                anything over *lookup*, which is the honest question for a
                brand whose replies are highly templated.
"""
from __future__ import annotations
import re
from collections import Counter
from taxonomy import INTENTS, ALWAYS_ESCALATE, default_action

# Ordered: first match wins, so the specific patterns must precede the generic.
KEYWORD_RULES: list[tuple[str, str]] = [
    ("fraud_report",   r"\b(scam|phish|fraud|fake|counterfeit|impersonat|pretend(ing)? to be|"
                       r"identity|someone (else ?'?s?|is|has) (us(ing|ed)|order(ing|ed)) |"
                       r"using my (name|address|account|card)|unauthoris?zed (access|use))\b"),
    ("account_access", r"\b(locked|lock(ed)? out|suspend|deactivat|can'?t (log|sign) ?in|"
                       r"cannot (log|sign) ?in|reset (my )?password|password (isn'?t|not) work|2fa|"
                       r"verification code)\b"),
    ("billing_charge", r"\b(charged|charge|billed|cashback|gift ?card|balance|overcharg|"
                       r"double.?charg|amazon ?pay|wrong price|mrp|debited|deducted)\b"),
    ("refund_status",  r"\b(refund|money back|reimburse|credit(ed)? back|still waiting for (my )?money)\b"),
    ("return_replace", r"\b(return|replace|replacement|exchange|damaged|broken|faulty|defect|"
                       r"wrong (item|product|size)|missing (part|item)s?)\b"),
    ("order_change",   r"\b(cancel|change (the |my )?(address|delivery|slot|payment)|"
                       r"modify (my )?order|wrong address)\b"),
    ("delivery_not_received",
                       r"\b(not (been )?(deliver|receiv)|never (arriv|receiv|got)|marked as deliver|"
                       r"says deliver|shows deliver|lost|stolen|missing (package|parcel|order)|"
                       r"no(t)? (sign|trace) of|left (it )?(outside|in the rain|at the wrong))\b"),
    ("delivery_late",  r"\b(late|delay|still (waiting|not here|hasn'?t)|when will|eta|"
                       r"deliver(y|ed)? (date|today|tomorrow)|overdue|slow|expedite|"
                       r"one ?day delivery|next ?day)\b"),
    ("membership_prime",
                       r"\b(prime member|membership|subscription|auto.?renew|free trial|"
                       r"student prime|cancel prime|prime fee|annual fee)\b"),
    ("tech_support",   r"\b(app|website|site|kindle|fire ?tv|prime video|echo|alexa|playback|"
                       r"buffer|stream|error|crash|glitch|won'?t load|not working|sync|"
                       r"log ?in page)\b"),
    ("product_info",  r"\b(available|availability|compatib|will (it|this) work|does (it|this)|"
                       r"specs?|when (does|is) .{0,20}(sale|launch|release)|in stock|"
                       r"do you (sell|ship|deliver) to|how much|price of)\b"),
]
_COMPILED = [(name, re.compile(p, re.I)) for name, p in KEYWORD_RULES]

HUMAN_REQ = re.compile(
    r"\b(speak|talk) to (a|an) (human|person|manager|agent|supervisor)|"
    r"call me|callback|call back|real person|stop (the )?(bot|automated)|"
    r"escalate|complain(t)? to|legal action|lawyer|ombudsman|consumer (court|forum)|"
    r"bbc|press|media|journalist", re.I)
REPEAT = re.compile(
    r"\b(again|still|second time|third time|3rd time|2nd time|already (told|asked|sent|explained)|"
    r"as I said|no (one|body) (has )?(replied|responded|helped)|week(s)? (now|later)|"
    r"how many times|worst|pathetic|useless|disgusting|never (again|shopping)|cancel(ling)? my prime)\b",
    re.I)


class MajorityBaseline:
    """B0: the trivial baseline."""
    name = "B0_majority"

    def fit(self, intents):
        self.major = Counter(intents).most_common(1)[0][0]
        return self

    def handle(self, msg, prior_turn=""):
        # Always escalating is the cost-optimal trivial policy, not a strawman.
        return dict(intent=getattr(self, "major", "no_request"), action="escalate",
                    reply="", reason="trivial baseline: always escalate",
                    rules_fired=[], confidence=0.0)


class KeywordBaseline:
    """B1: rules only, no model."""
    name = "B1_keyword"

    def handle(self, msg, prior_turn=""):
        intent = next((n for n, rx in _COMPILED if rx.search(msg)), "no_request")
        action = default_action(intent)
        fired = []
        if intent in ALWAYS_ESCALATE:
            fired.append("E1_account_or_pii")
        if HUMAN_REQ.search(msg):
            action, _ = "escalate", fired.append("E4_human_requested")
        if REPEAT.search(msg):
            action, _ = "escalate", fired.append("E5_repeat_or_hostile")
        return dict(intent=intent, action=action, reply="", confidence=1.0,
                    reason="keyword rule match", rules_fired=fired)


class RetrievalBaseline:
    """B2: classify by nearest neighbour's label, reply by copying its reply."""
    name = "B2_retrieval"

    def __init__(self, retriever, labels: dict[str, str] | None = None):
        self.r = retriever
        self.labels = labels or {}
        self.kw = KeywordBaseline()

    def handle(self, msg, prior_turn=""):
        hits = self.r.search(msg, k=3)
        top = hits[0] if hits else None
        # No labels exist for the corpus, so fall back to the keyword rules for
        # the intent and let this baseline be tested on what it uniquely does:
        # producing a reply by lookup instead of generation.
        intent = self.labels.get(top["case_id"]) if top else None
        if intent not in INTENTS:
            intent = self.kw.handle(msg)["intent"]
        return dict(intent=intent, action=default_action(intent),
                    reply=top["brand_reply"] if top else "", confidence=1.0,
                    reason="copied nearest historical reply",
                    rules_fired=["E1_account_or_pii"] if intent in ALWAYS_ESCALATE else [],
                    top_similarity=top["score"] if top else 0.0)


def demo():
    kw = KeywordBaseline()
    assert kw.handle("I want a refund for my order")["intent"] == "refund_status"
    assert kw.handle("my parcel says delivered but I never got it")["intent"] == "delivery_not_received"
    assert kw.handle("someone is using my address to order things")["intent"] == "fraud_report"
    assert kw.handle("how do I cancel my prime membership")["intent"] == "order_change"  # 'cancel' wins first
    assert kw.handle("lovely weather today")["intent"] == "no_request"
    # escalation triggers are independent of intent
    r = kw.handle("when will it arrive? this is the third time I have asked")
    assert r["action"] == "escalate" and "E5_repeat_or_hostile" in r["rules_fired"]
    r = kw.handle("is this available in India? let me speak to a human")
    assert r["action"] == "escalate" and "E4_human_requested" in r["rules_fired"]
    assert MajorityBaseline().fit(["a", "b", "a"]).handle("x")["intent"] == "a"
    print("baselines ok")


if __name__ == "__main__":
    demo()
