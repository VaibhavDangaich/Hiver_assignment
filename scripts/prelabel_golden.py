"""Pre-annotate the golden slices, to be adjudicated by hand afterwards.

Deliberate asymmetry: pre-annotation uses the STRONGER model (sonnet) and sees
the brand's actual historical reply as a hint. The system under test uses the
WEAKER model (haiku) and never sees that reply. So the reference is not produced
by the system being graded.

This is pre-annotation, not labelling. `data/golden/labelled_*.jsonl` is only
written by scripts/adjudicate.py, which records every human change.
"""
import concurrent.futures as cf, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from llm import LLM, extract_json, JUDGE_MODEL
from taxonomy import INTENTS, ESCALATION_RULES

SYS = "You are an expert support-operations analyst annotating a gold standard. Output JSON only."

TMPL = """Annotate this customer message sent to Amazon's support account on Twitter.

INTENT TAXONOMY (choose exactly one):
{intents}

ESCALATION RULES (an escalation is required if ANY rule applies):
{rules}
If no rule applies, the message can be auto-handled.

The brand's actual historical reply is shown only to help you infer what the
customer was really asking. Do NOT assume the brand handled it correctly.

CUSTOMER MESSAGE: {msg}
{ctx}ACTUAL HISTORICAL BRAND REPLY: {reply}

Return JSON:
{{"intent": "<one key>",
  "action": "auto" | "escalate",
  "rules_fired": ["E1_account_or_pii", ...],
  "reason": "<one sentence, why this action>",
  "ambiguous": true | false,
  "notes": "<anything a human adjudicator should check, else empty>"}}"""


def main() -> None:
    llm = LLM()
    intents = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    rules = "\n".join(f"- {k}: {v}" for k, v in ESCALATION_RULES.items())

    rows = []
    for sl in ("random", "enriched"):
        rows += [json.loads(l) for l in open(f"data/golden/unlabelled_{sl}.jsonl")]

    def annotate(r):
        ctx = f"EARLIER IN THREAD: {r['prior_turn']}\n" if r.get("prior_turn") else ""
        p = TMPL.format(intents=intents, rules=rules, msg=r["customer_msg"],
                        ctx=ctx, reply=r["brand_reply"])
        for _ in range(3):
            try:
                d = extract_json(llm.complete(p, system=SYS, model=JUDGE_MODEL, max_tokens=700))
                if d.get("intent") in INTENTS and d.get("action") in ("auto", "escalate"):
                    return {**r, "pre": d}
            except Exception as e:
                err = str(e)[:120]
        return {**r, "pre": {"intent": "no_request", "action": "escalate",
                             "rules_fired": [], "reason": "PRE-ANNOTATION FAILED",
                             "ambiguous": True, "notes": "needs full human pass"}}

    with cf.ThreadPoolExecutor(10) as ex:
        out = list(ex.map(annotate, rows))

    with open("data/golden/preannotated.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    import collections
    print("intents:", collections.Counter(r["pre"]["intent"] for r in out).most_common(), file=sys.stderr)
    print("actions:", collections.Counter(r["pre"]["action"] for r in out), file=sys.stderr)
    print("ambiguous:", sum(bool(r["pre"].get("ambiguous")) for r in out), file=sys.stderr)
    print("llm:", llm.stats(), file=sys.stderr)


if __name__ == "__main__":
    main()
