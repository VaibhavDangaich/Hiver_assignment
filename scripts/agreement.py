"""Judge-vs-human agreement, and label-quality evidence for the gold set.

The assignment asks for "evidence of how well your judge agrees with a human".
A judge that has never been checked against a person is decoration, so this
computes, on the overlap between artifacts/results/human_reply_ratings.jsonl and
the judge's scores:

  * exact and within-1 agreement per axis on the 1-5 scales
  * Spearman rho on the composite (rank agreement is the honest question: we
    care whether the judge ORDERS replies like a human, not whether it picks
    the same integer)
  * Cohen's kappa on the unsupported_claim binary, which is the axis with real
    consequences, plus the confusion counts behind it

It also reports gold-label provenance: how many items a human actually changed
during adjudication, which bounds how much of the gold set is human judgement.
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from judge import AXES

RES = pathlib.Path("artifacts/results")


def spearman(a, b):
    """Rank correlation with average ranks for ties; no scipy needed here."""
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def kappa(a, b):
    """Cohen's kappa for two binary raters."""
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    out = {}

    # ---- gold label provenance -------------------------------------------
    gp = pathlib.Path("data/golden/labelled.jsonl")
    if gp.exists():
        gold = [json.loads(l) for l in gp.open()]
        changed = sum(1 for g in gold if g.get("human_changed"))
        out["gold_provenance"] = dict(
            n=len(gold), human_changed=changed,
            human_changed_rate=round(changed / len(gold), 3) if gold else 0.0,
            by_queue_tag={t: sum(1 for g in gold if g.get("queue_tag") == t)
                          for t in ("CONTESTED", "ambiguous", "routine")},
            note="human_changed counts items where the adjudicator overrode the "
                 "machine proposal; the rest were confirmed by a human, not unreviewed.")
    else:
        out["gold_provenance"] = dict(
            n=0, note="data/golden/labelled.jsonl absent -- run scripts/adjudicate.py. "
                      "Until then all metrics use the A/B machine consensus and are "
                      "labelled machine_consensus_provisional.")

    # ---- pass A/B agreement (label difficulty evidence) -------------------
    pre = [json.loads(l) for l in open("data/golden/preannotated.jsonl")]
    both = [r for r in pre if r.get("pre_b", {}).get("intent")]
    if both:
        ai = [r["pre"]["intent"] for r in both]
        bi = [r["pre_b"]["intent"] for r in both]
        aa = [1 if r["pre"]["action"] == "escalate" else 0 for r in both]
        ba = [1 if r["pre_b"]["action"] == "escalate" else 0 for r in both]
        out["machine_pass_agreement"] = dict(
            n=len(both),
            intent_agreement=round(sum(x == y for x, y in zip(ai, bi)) / len(both), 3),
            action_agreement=round(sum(x == y for x, y in zip(aa, ba)) / len(both), 3),
            action_kappa=round(kappa(aa, ba), 3),
            note="two independent pre-annotation passes; B never saw the brand's reply.")

    # ---- judge vs human ---------------------------------------------------
    hp = RES / "human_reply_ratings.jsonl"
    if hp.exists() and (RES / "predictions.jsonl").exists():
        human = {(json.loads(l)["case_id"], json.loads(l)["system"]): json.loads(l)
                 for l in hp.open()}
        preds = {(p["case_id"], p["system"]): p
                 for p in (json.loads(l) for l in open(RES / "predictions.jsonl"))}
        pairs = [(h, preds[k]["judge"]) for k, h in human.items()
                 if k in preds and preds[k].get("judge", {}).get("scored")]
        if pairs:
            ax = {}
            for a in AXES:
                hv = [h[a] for h, _ in pairs]
                jv = [j[a] for _, j in pairs]
                ax[a] = dict(
                    exact=round(sum(x == y for x, y in zip(hv, jv)) / len(pairs), 3),
                    within_1=round(sum(abs(x - y) <= 1 for x, y in zip(hv, jv)) / len(pairs), 3),
                    mean_human=round(sum(hv) / len(hv), 2),
                    mean_judge=round(sum(jv) / len(jv), 2),
                    judge_bias=round((sum(jv) - sum(hv)) / len(hv), 2),
                    spearman=round(spearman(hv, jv), 3))
            hc = [1 if h["unsupported_claim"] else 0 for h, _ in pairs]
            jc = [1 if j["unsupported_claim"] else 0 for _, j in pairs]
            out["judge_vs_human"] = dict(
                n=len(pairs), per_axis=ax,
                composite_spearman=round(spearman([h["composite"] for h, _ in pairs],
                                                  [j["composite"] for _, j in pairs]), 3),
                unsupported_claim=dict(
                    agreement=round(sum(x == y for x, y in zip(hc, jc)) / len(pairs), 3),
                    cohens_kappa=round(kappa(hc, jc), 3),
                    human_flagged=sum(hc), judge_flagged=sum(jc),
                    judge_missed=sum(1 for x, y in zip(hc, jc) if x and not y),
                    judge_false_alarm=sum(1 for x, y in zip(hc, jc) if y and not x)))
        else:
            out["judge_vs_human"] = dict(n=0, note="no overlap between human ratings and judged predictions")
    else:
        out["judge_vs_human"] = dict(
            n=0, note="artifacts/results/human_reply_ratings.jsonl absent -- "
                      "run scripts/rate_replies.py. Judge scores are UNVALIDATED until then.")

    (RES / "agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {RES / 'agreement.json'}")


def demo():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 2, 2, 3], [1, 2, 2, 3]) - 1.0) < 1e-9   # ties
    assert abs(kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert abs(kappa([1, 0, 1, 0], [0, 1, 0, 1]) + 1.0) < 1e-9
    assert kappa([], []) == 0.0
    print("agreement math ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
