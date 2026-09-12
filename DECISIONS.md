# Decision log

Non-obvious calls, and why. Reverse-chronological within themes, not by importance.

### Data and brand

1. **Picked AmazonHelp on measured deflection rate, not tweet volume.**
   The obvious picks are the biggest accounts. I scored the top 40 brands by the
   share of replies that just push the customer to DM/phone/email, because a brand
   that deflects has no *resolutions* to ground replies in. `AppleSupport` deflects
   **53%** of replies and `TMobileHelp` **82%**; `AmazonHelp` deflects **5.3%** while
   still leaving 168k usable reply pairs. Choosing Apple on volume would have
   produced an agent that learned to say "please DM us".

2. **Kept the dataset out of git, with a verifying fetcher.**
   The raw CSV is 516MB. `scripts/fetch_data.py` pulls a public mirror and checks
   byte count and header before anything downstream runs, so a silently truncated
   download fails loudly instead of skewing results.

3. **Used a Hugging Face mirror rather than the Kaggle original.**
   Kaggle needs credentials the grader may not have. Same file, verified by size
   and header, provenance cited in the README. Reproducibility beat source purity.

4. **Temporal split, partitioned by customer, not a random split.**
   Retrieval over a random split leaks: the same customer's own thread can be
   retrieved as "precedent" for their own message. The corpus is the earlier 70%
   of the timeline and the golden set is drawn from later traffic, with every
   customer forced entirely onto one side. `build_corpus.py` asserts no customer
   and no case id crosses over. This costs a few accuracy points and is the main
   reason the numbers are believable.

5. **Precision-oriented language filter, audited by hand, not trusted.**
   30% of AmazonHelp pairs are non-English. Rather than add a langid dependency I
   wrote a 15-line heuristic — then read 90 samples of its decisions to find out
   how wrong it is (`notes/lang_audit.md`): **100% precision, ~20% recall loss**,
   and every miss is a short fragment like "still not here". I kept it *and*
   documented the resulting bias, because the corpus is now skewed toward long
   self-contained messages and that inflates measured performance.

6. **Scrubbed PII at ingest, not at display.**
   Real order IDs, phone numbers and emails are live in this data. `textnorm.py`
   replaces them before anything is stored, so no prompt, cache entry, or committed
   artifact carries them. Order-ID patterns are matched *before* phone patterns
   because the digit runs collide.

### Taxonomy and policy

7. **Wrote the taxonomy and escalation rules before labelling anything.**
   If you label first, "the rule" becomes whatever you did — unfalsifiable. Both
   live in `src/taxonomy.py`, fixed in a commit that precedes every label.

8. **Induced intents from the data in two passes, and threw away the first.**
   KMeans over 76k messages was only good enough to scout the topic spread — its
   biggest cluster held 36% of traffic and mixed refunds with rants, so it was
   unusable as a taxonomy. An LLM open-coding pass over a 240-message sample
   naming *the request* (not the topic) produced 187 free-text codes that collapse
   cleanly into 11 actionable families.

9. **Made `no_request` a first-class intent.**
   22% of sampled traffic wants nothing actionable — venting, jokes, praise,
   fragments. Most taxonomies bury this in "other" and then report accuracy on a
   problem that does not exist. A production router must decide what to do with a
   fifth of its traffic.

10. **Declared an explicit 10:1 cost model, in code, before seeing results.**
    Accuracy treats both routing errors as equal; they are not. Auto-replying to
    something that needed a human is a public, reputational error. A needless
    escalation costs one agent-minute. Constants live in `taxonomy.py` so the
    metric cannot be tuned after the fact.

11. **Some intents can never be auto-handled, regardless of model confidence.**
    Refunds, account access, billing, returns, lost packages and fraud are
    `ALWAYS_ESCALATE`. A confident model is not an authorised one. This is a
    policy statement, not a prediction, so it is enforced deterministically in
    `agent.py:_enforce` rather than requested in the prompt.

### System

12. **One LLM call for intent + routing + reply, not three.**
    They are not independent: what you may safely say depends on whether a human
    must act. Separate calls contradict each other and cost 3×. The guardrail layer
    then overrides the model where policy is absolute.

13. **Deterministic guardrails that can only ever escalate.**
    Grounding floor (top retrieval similarity < 0.13), confidence floor, policy
    intents, self-contradiction, and a regex that catches drafts promising money,
    replacements or delivery dates. Every one is unit-tested with a fake LLM, so
    the safety behaviour is proven without a network call.

14. **TF-IDF retrieval instead of embeddings.**
    On 76k short, typo-heavy, jargon-dense tweets, lexical overlap carries most of
    the signal. It is free, needs no API key, and is *deterministic* — the grader
    reruns it and gets identical results. Word n-grams catch phrasing; char n-grams
    survive the typos. Embeddings are the obvious next upgrade, not a prerequisite.

15. **Did not train a supervised classifier.**
    Tempting as a third baseline, but there are only ~220 labels across 12 classes
    (~18 each) and they *are* the evaluation set. Training on them would either
    destroy the eval or produce a classifier too weak to be a fair comparison.

16. **No agent framework (LangGraph et al.).**
    The flow is linear — classify → retrieve → draft → route — with one model call
    and no cycles, persisted state, or human-in-the-loop interrupts. A graph runtime
    would add install surface that threatens the 15-minute reproduction target and
    would make the deterministic guardrails harder to prove, buying nothing.

### Evaluation

17. **The judge is a different, stronger model than the generator.**
    Haiku drafts; Sonnet judges. A model grading its own output is a known failure
    mode. The judge also scores against the brand's *actual* historical reply, so
    "good" means brand-consistent rather than generically polite — a fluent reply
    promising what Amazon never promises scores badly.

18. **Two independent machine pre-annotation passes, then a human.**
    Pass A sees the brand's real reply; pass B does not and asks the routing
    question first. They agreed on intent **85.0%** and routing **90.5%** (n=220).
    Disagreements are queued to a human first, so human attention goes where the
    label is genuinely contested instead of being spread evenly over easy items.

19. **A one-line binary (`unsupported_claim`) alongside the 1–5 scales.**
    Models are far more reliable on "did it state something unsupported, yes/no"
    than on folding hallucination risk into a quality score. It is also the axis
    with real consequences, so it gets Cohen's κ against a human rather than a mean.

20. **Never averaged the random and enriched slices.**
    The enriched slice over-samples fraud, lockouts and billing on purpose, to get
    enough positives per class for per-class metrics to mean anything. Blending it
    into one headline number would flatter macro-F1 while describing traffic that
    does not exist. Every table reports them separately.

21. **Kept the trivial baseline honest instead of making it a strawman.**
    B0 always escalates. Under the declared cost model that is a genuinely strong
    policy — it never makes the expensive error. It turns out to be hard to beat on
    cost, which is the single most useful fact in the report and would have been
    hidden by a deliberately weak baseline.

22. **Committed the LLM cache as the reproduction artifact.**
    `make reproduce` replays it with `LLM_OFFLINE=1` — no API key, no network, and a
    cache miss raises rather than silently calling out. The alternative, asking a
    grader to spend money and get different numbers, is not reproduction.
