"""Run every system over the golden set and score the replies.

Systems: B0 majority, B1 keyword, B2 retrieval-copy, AGENT.
Writes artifacts/results/predictions.jsonl -- one row per (case, system).

Gold labels come from data/golden/labelled.jsonl (human-adjudicated). If that
file is absent the run falls back to the A/B machine consensus and stamps every
row `gold_source="machine_consensus_provisional"`, so a number produced without
human labels can never be mistaken for one produced with them.
"""
import argparse, concurrent.futures as cf, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from agent import SupportAgent
from baselines import MajorityBaseline, KeywordBaseline, RetrievalBaseline
from retrieval import CaseRetriever
from judge import ReplyJudge
from llm import LLM

RES = pathlib.Path("artifacts/results")


def load_gold() -> tuple[list[dict], str]:
    gold_path = pathlib.Path("data/golden/labelled.jsonl")
    pre = {json.loads(l)["case_id"]: json.loads(l)
           for l in open("data/golden/preannotated.jsonl")}
    if gold_path.exists():
        rows = [json.loads(l) for l in gold_path.open()]
        if rows:
            for r in rows:                       # carry thread context through
                r.setdefault("prior_turn", pre.get(r["case_id"], {}).get("prior_turn", ""))
            return rows, "human_adjudicated"
    rows = []
    for r in pre.values():
        a, b = r["pre"], r["pre_b"]
        if a["intent"] != b["intent"] or a["action"] != b["action"]:
            continue                              # contested items need a human
        rows.append(dict(case_id=r["case_id"], slice=r["slice"],
                         customer_msg=r["customer_msg"], prior_turn=r.get("prior_turn", ""),
                         brand_reply=r["brand_reply"],
                         gold_intent=a["intent"], gold_action=a["action"]))
    return rows, "machine_consensus_provisional"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--systems", default="", help="comma-separated subset, e.g. B0_majority,B1_keyword")
    args = ap.parse_args()

    gold, source = load_gold()
    if args.limit:
        gold = gold[:args.limit]
    print(f"gold: {len(gold)} items, source={source}", file=sys.stderr)

    llm = LLM()
    retr = CaseRetriever()
    systems = {
        "B0_majority":  MajorityBaseline().fit([g["gold_intent"] for g in gold]),
        "B1_keyword":   KeywordBaseline(),
        "B2_retrieval": RetrievalBaseline(retr),
        "AGENT":        SupportAgent(retr, llm),
    }

    def run_one(job):
        name, sysm, g = job
        try:
            out = sysm.handle(g["customer_msg"], g.get("prior_turn", ""))
        except Exception as e:
            print(f"  {name} failed on {g['case_id']}: {str(e)[:120]}", file=sys.stderr)
            out = dict(intent="no_request", action="escalate", reply="",
                       confidence=0.0, rules_fired=[], reason=f"ERROR {str(e)[:80]}")
        return dict(case_id=g["case_id"], slice=g["slice"], system=name,
                    customer_msg=g["customer_msg"], prior_turn=g.get("prior_turn", ""),
                    brand_reply=g["brand_reply"],
                    gold_intent=g["gold_intent"], gold_action=g["gold_action"],
                    gold_source=source,
                    pred_intent=out.get("intent"), pred_action=out.get("action"),
                    reply=out.get("reply", ""), confidence=out.get("confidence", 0.0),
                    rules_fired=out.get("rules_fired", []),
                    reason=out.get("reason", ""),
                    guardrail_reasons=out.get("guardrail_reasons", []),
                    top_similarity=out.get("top_similarity", 0.0),
                    retrieved=out.get("retrieved", []))

    if args.systems:
        want = {x.strip() for x in args.systems.split(",")}
        missing = want - set(systems)
        assert not missing, f"unknown systems: {missing}"
        systems = {k: v for k, v in systems.items() if k in want}
    jobs = [(n, s, g) for g in gold for n, s in systems.items()]
    # Only the agent hits the network; baselines are instant.
    with cf.ThreadPoolExecutor(args.workers) as ex:
        preds = list(ex.map(run_one, jobs))
    print(f"predictions: {len(preds)}  llm={llm.stats()}", file=sys.stderr)

    if not args.no_judge:
        judge = ReplyJudge(llm)
        to_score = [p for p in preds if (p.get("reply") or "").strip()]
        print(f"judging {len(to_score)} non-empty replies", file=sys.stderr)

        def score(p):
            try:
                return p, judge.score(p["customer_msg"], p["reply"],
                                      p["brand_reply"], p.get("prior_turn", ""))
            except Exception as e:
                return p, dict(scored=False, reason=str(e)[:120])

        with cf.ThreadPoolExecutor(args.workers) as ex:
            for p, s in ex.map(score, to_score):
                p["judge"] = s
        print(f"judged. llm={llm.stats()}", file=sys.stderr)

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {RES / 'predictions.jsonl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
