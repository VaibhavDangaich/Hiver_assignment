"""Compute every headline number from artifacts/results/predictions.jsonl.

Reports, per system and per slice:
  * intent: accuracy, macro-F1 (macro because rare intents are the risky ones),
    and the confusion pairs that actually drive the error
  * routing: accuracy, plus the two error types named separately because they
    are not equally bad -- missed escalation (auto'd something that needed a
    human) vs needless escalation
  * expected cost under the 10:1 model declared in taxonomy.py
  * auto-rate and, on auto-handled cases only, the judge's quality scores

No metric is averaged across the random and enriched slices: the enriched slice
is deliberately not representative, so a blended number would be meaningless.
"""
import collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from taxonomy import INTENTS, COST_MISSED_ESCALATION, COST_NEEDLESS_ESCALATION
from judge import AXES

RES = pathlib.Path("artifacts/results")


def prf(rows, labels):
    """Per-class precision/recall/F1 + macro-F1, computed explicitly so the
    denominators are auditable rather than hidden in a library call."""
    out, f1s = {}, []
    for lab in labels:
        tp = sum(1 for r in rows if r["pred_intent"] == lab and r["gold_intent"] == lab)
        fp = sum(1 for r in rows if r["pred_intent"] == lab and r["gold_intent"] != lab)
        fn = sum(1 for r in rows if r["pred_intent"] != lab and r["gold_intent"] == lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        support = tp + fn
        if support:                     # classes absent from gold don't dilute macro-F1
            f1s.append(f1)
        out[lab] = dict(precision=round(p, 3), recall=round(rc, 3),
                        f1=round(f1, 3), support=support)
    return out, (sum(f1s) / len(f1s) if f1s else 0.0)


def routing(rows):
    missed = sum(1 for r in rows if r["pred_action"] == "auto" and r["gold_action"] == "escalate")
    needless = sum(1 for r in rows if r["pred_action"] == "escalate" and r["gold_action"] == "auto")
    correct = sum(1 for r in rows if r["pred_action"] == r["gold_action"])
    n = len(rows)
    gold_auto = sum(1 for r in rows if r["gold_action"] == "auto")
    return dict(
        n=n, accuracy=round(correct / n, 3) if n else 0.0,
        missed_escalation=missed,
        missed_escalation_rate=round(missed / n, 3) if n else 0.0,
        needless_escalation=needless,
        needless_escalation_rate=round(needless / n, 3) if n else 0.0,
        auto_rate=round(sum(1 for r in rows if r["pred_action"] == "auto") / n, 3) if n else 0.0,
        gold_auto_rate=round(gold_auto / n, 3) if n else 0.0,
        # Of everything we auto-handled, how much should have gone to a human?
        auto_precision=round(
            sum(1 for r in rows if r["pred_action"] == "auto" and r["gold_action"] == "auto")
            / max(1, sum(1 for r in rows if r["pred_action"] == "auto")), 3),
        expected_cost=round(
            (missed * COST_MISSED_ESCALATION + needless * COST_NEEDLESS_ESCALATION) / n, 3)
        if n else 0.0,
    )


def judged(rows):
    sc = [r["judge"] for r in rows if r.get("judge", {}).get("scored")]
    if not sc:
        return dict(n_judged=0)
    out = {f"mean_{a}": round(sum(s[a] for s in sc) / len(sc), 3) for a in AXES}
    out["n_judged"] = len(sc)
    out["mean_composite"] = round(sum(s["composite"] for s in sc) / len(sc), 3)
    out["unsupported_claim_rate"] = round(
        sum(1 for s in sc if s["unsupported_claim"]) / len(sc), 3)
    out["worst_axis_counts"] = dict(collections.Counter(
        s["worst_axis"] for s in sc if s.get("worst_axis")))
    return out


def main() -> None:
    rows = [json.loads(l) for l in open(RES / "predictions.jsonl")]
    source = rows[0].get("gold_source", "unknown") if rows else "unknown"
    systems = sorted({r["system"] for r in rows})
    report = dict(gold_source=source, n_cases=len({r["case_id"] for r in rows}), systems={})

    for s in systems:
        srows = [r for r in rows if r["system"] == s]
        entry = {}
        for sl in ("all", "random", "enriched"):
            sub = srows if sl == "all" else [r for r in srows if r["slice"] == sl]
            if not sub:
                continue
            per_class, macro = prf(sub, list(INTENTS))
            acc = sum(1 for r in sub if r["pred_intent"] == r["gold_intent"]) / len(sub)
            e = dict(n=len(sub), intent_accuracy=round(acc, 3),
                     intent_macro_f1=round(macro, 3), routing=routing(sub))
            # Reply quality only counts where the system actually auto-posts.
            e["reply_quality_auto_only"] = judged([r for r in sub if r["pred_action"] == "auto"])
            e["reply_quality_all_drafts"] = judged(sub)
            if sl == "all":
                e["per_class"] = per_class
                e["top_confusions"] = collections.Counter(
                    (r["gold_intent"], r["pred_intent"]) for r in sub
                    if r["gold_intent"] != r["pred_intent"]).most_common(8)
            entry[sl] = e
        report["systems"][s] = entry

    (RES / "metrics.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- console summary ----
    print(f"\ngold_source = {source}   cases = {report['n_cases']}")
    for sl in ("random", "enriched"):
        print(f"\n=== slice: {sl} ===")
        hdr = f"{'system':14s} {'n':>4s} {'int.acc':>8s} {'macroF1':>8s} {'route.acc':>9s} " \
              f"{'missed':>7s} {'needless':>9s} {'auto%':>6s} {'cost':>6s} {'quality':>8s}"
        print(hdr); print("-" * len(hdr))
        for s in systems:
            e = report["systems"][s].get(sl)
            if not e:
                continue
            r, q = e["routing"], e["reply_quality_auto_only"]
            qs = f"{q['mean_composite']:.2f}" if q.get("n_judged") else "-"
            print(f"{s:14s} {e['n']:4d} {e['intent_accuracy']:8.3f} {e['intent_macro_f1']:8.3f} "
                  f"{r['accuracy']:9.3f} {r['missed_escalation']:7d} {r['needless_escalation']:9d} "
                  f"{r['auto_rate']*100:5.1f}% {r['expected_cost']:6.2f} {qs:>8s}")
    print(f"\nwrote {RES / 'metrics.json'}")
    print("cost = expected cost per case "
          f"({COST_MISSED_ESCALATION:.0f}x missed escalation, "
          f"{COST_NEEDLESS_ESCALATION:.0f}x needless). Lower is better.")


if __name__ == "__main__":
    main()
