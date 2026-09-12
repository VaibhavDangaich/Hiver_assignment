"""Single source of truth for intents and the escalation policy.

Both were fixed BEFORE any example was labelled, so label disagreements are
analysable against a written rule rather than against recollected intuition.

Provenance of the intent set: TF-IDF+KMeans scouting over 76,181 corpus messages
(k=14) to find the topical spread, then an LLM open-coding pass over a random
240-message sample naming the *request* in each ("what does the customer want",
not the topic). The 187 free-text codes collapsed into the 11 actionable
families below; 22% of the sample wanted nothing actionable, which is why
`no_request` is a first-class label and not a dustbin.
"""

INTENTS: dict[str, str] = {
    "delivery_late":
        "Order is confirmed and en route but late, slower than promised, or the "
        "customer wants it expedited / wants a firm date.",
    "delivery_not_received":
        "Tracking claims delivered (or a delivery was attempted) but the customer "
        "does not have the package: marked-delivered-not-received, lost, stolen, "
        "left in a wrong or unsafe place, failed attempt.",
    "refund_status":
        "A refund was promised, owed, or already initiated and the customer is "
        "chasing it, disputing its amount, or asking when it lands.",
    "return_replace":
        "Customer wants to send something back or get it replaced: faulty, damaged, "
        "wrong, counterfeit, or unwanted item; or asks how the return works.",
    "order_change":
        "Customer wants to alter or stop an order already placed: cancel, change "
        "delivery address or slot, change payment method.",
    "account_access":
        "Customer cannot get into their account: locked, suspended, password or "
        "registered-email reset failing, 2FA problems.",
    "billing_charge":
        "A money movement the customer disputes or cannot explain: unauthorised or "
        "duplicate charge, wrong price, missing cashback or gift-card balance, "
        "Amazon Pay failure.",
    "membership_prime":
        "Prime as a subscription: what it includes, joining, cancelling, "
        "auto-renewal, student/trial terms. Not a disputed Prime charge -> billing_charge.",
    "product_info":
        "Pre-purchase or general question answerable without opening an account: "
        "availability, compatibility, specs, sale dates, whether a service exists "
        "in a region.",
    "tech_support":
        "An Amazon-operated digital surface is broken: app, website, Kindle, Fire "
        "TV, Prime Video playback, sync, login pages that error out.",
    "fraud_report":
        "Customer reports criminal or abusive activity rather than asking for their "
        "own order to be fixed: phishing calls/emails impersonating Amazon, "
        "counterfeit sellers, someone else using their identity or address.",
    "no_request":
        "Nothing actionable: venting with no specific ask, praise, jokes, "
        "commentary, off-topic mentions, or a fragment too vague to act on.",
}

# ---------------------------------------------------------------------------
# Escalation policy. Written before labelling. Any rule firing => escalate.
# ---------------------------------------------------------------------------
ESCALATION_RULES: dict[str, str] = {
    "E1_account_or_pii":
        "Resolving it requires looking at, verifying, or changing account-specific "
        "data (order records, payment instruments, addresses, identity). A public "
        "reply cannot do this safely.",
    "E2_money_commitment":
        "A satisfactory answer commits money or goods: issuing a refund, "
        "compensation, a free replacement, or a credit.",
    "E3_safety_legal_fraud":
        "Fraud, phishing, counterfeit goods, identity misuse, threats of legal "
        "action, regulator complaints, or any physical-safety or harassment claim.",
    "E4_human_requested":
        "The customer explicitly asks for a human, a manager, a callback, or says "
        "an automated/scripted reply already failed them.",
    "E5_repeat_or_hostile":
        "Evidence the issue already failed to be resolved (second/third attempt, "
        "'still waiting', prior agent named) or the customer is abusive, "
        "threatening to churn, or publicly escalating with press/social pressure.",
    "E6_no_grounding":
        "Retrieval surfaced no sufficiently similar historical case, so any reply "
        "would be improvised rather than grounded in how this brand actually acts.",
    "E7_low_confidence":
        "The classifier is not confident which intent applies, so the reply would "
        "be built on a guess.",
}

# Default routing per intent, applied when no rule above fires.
# AUTO means "post this reply without a human reading it first".
AUTO_ELIGIBLE: set[str] = {
    "delivery_late",      # set expectations + point at the tracking self-serve flow
    "membership_prime",   # documented policy, same answer every time
    "product_info",       # public information
    "tech_support",       # generic troubleshooting first line
    "no_request",         # acknowledge or stay silent; nothing is promised
}
ALWAYS_ESCALATE: set[str] = {
    "delivery_not_received",  # E1+E2: needs the order record, usually ends in money
    "refund_status",          # E1+E2
    "return_replace",         # E1+E2
    "order_change",           # E1: mutates the account
    "account_access",         # E1: never do identity work in public
    "billing_charge",         # E1+E2
    "fraud_report",           # E3
}

ACTIONS = ("auto", "escalate")

# Cost model. Justified in the report; stated here so the metric cannot drift.
# A wrongly auto-handled case that needed a human is the expensive error: the
# customer is publicly told something wrong or is silently dropped, and Amazon
# support on Twitter is a reputational surface. A needless escalation costs one
# agent-minute.
COST_MISSED_ESCALATION = 10.0   # predicted auto, should have been escalate
COST_NEEDLESS_ESCALATION = 1.0  # predicted escalate, could have been auto


def default_action(intent: str) -> str:
    return "escalate" if intent in ALWAYS_ESCALATE else "auto"


def demo():
    assert set(AUTO_ELIGIBLE) | set(ALWAYS_ESCALATE) == set(INTENTS), \
        "every intent needs a default route"
    assert not (AUTO_ELIGIBLE & ALWAYS_ESCALATE), "an intent cannot be both"
    assert default_action("refund_status") == "escalate"
    assert default_action("product_info") == "auto"
    assert all(len(v) > 40 for v in INTENTS.values()), "definitions must be usable by a labeller"
    print(f"taxonomy ok: {len(INTENTS)} intents, {len(ESCALATION_RULES)} escalation rules")


if __name__ == "__main__":
    demo()
