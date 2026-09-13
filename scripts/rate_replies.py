"""Human reply-rating CLI -> produces the data for judge-vs-human agreement.

The assignment asks for "evidence of how well your judge agrees with a human".
That evidence cannot come from the judge. So: a human scores a blinded sample of
drafts on the SAME anchored 1-5 axes and the SAME unsupported_claim binary, and
scripts/agreement.py compares the two.

Blinding: replies are shown in shuffled order with the producing system hidden,
so the rater cannot favour the agent over a baseline.

Usage: python scripts/rate_replies.py --n 60
Keys per axis: 1-5 | u toggle unsupported-claim | s skip | q save+quit
"""
import argparse, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from judge import AXES

OUT = pathlib.Path("artifacts/results/human_reply_ratings.jsonl")
ANCHORS = {
    "grounded": "5 matches reference approach, no invented facts | 3 adds an unsupported step | 1 invents policy/offers",
    "helpful":  "5 concrete next step for THIS request | 3 relevant but generic | 1 misreads or filler",
    "tone":     "5 warm, brief, non-defensive like the reference | 3 stiff or over-apologetic | 1 robotic or dismissive",
    "safe":     "5 nothing risky | 3 vague commitment | 1 PII, promises money/goods, or costly advice",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--preds", default="artifacts/results/predictions.jsonl")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.preds)]
    # Only items the judge actually scored: agreement.py joins on (case_id,
    # system), so an item with no judge score can never contribute evidence
    # and would just burn a human's time for nothing.
    items = [r for r in rows if (r.get("reply") or "").strip()
             and r.get("judge", {}).get("scored")]
    rng = random.Random(args.seed)
    # Stratify by system so a plain shuffle can't accidentally hand the human
    # mostly one system's replies -- agreement needs every system represented.
    by_sys: dict[str, list] = {}
    for r in items:
        by_sys.setdefault(r["system"], []).append(r)
    for v in by_sys.values():
        rng.shuffle(v)
    per = max(1, args.n // max(1, len(by_sys)))
    items = [r for v in by_sys.values() for r in v[:per]]
    rng.shuffle(items)                      # blinding: system order is randomised
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for l in OUT.open():
            try:
                d = json.loads(l); done.add((d["case_id"], d["system"]))
            except json.JSONDecodeError:
                continue
    todo = [r for r in items if (r["case_id"], r["system"]) not in done][:args.n]

    print(f"\nRating {len(todo)} replies ({len(done)} already done). "
          f"System identity is hidden.\n" + "=" * 78)
    fh = OUT.open("a")
    try:
        for n, r in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}]")
            print(f"  CUSTOMER: {r['customer_msg']}")
            if r.get("prior_turn"):
                print(f"  prior:    {r['prior_turn'][:150]}")
            print(f"  REFERENCE (what Amazon actually said): {r['brand_reply'][:200]}")
            print(f"  DRAFT:    {r['reply']}")
            scores, unsupported, skip = {}, False, False
            for ax in AXES:
                print(f"    {ax}: {ANCHORS[ax]}")
                while True:
                    s = input(f"    {ax} [1-5] ").strip().lower()
                    if s == "q":
                        print(f"saved {n - 1}"); fh.close(); return
                    if s == "s":
                        skip = True; break
                    if s == "u":
                        unsupported = not unsupported
                        print(f"      unsupported_claim -> {unsupported}"); continue
                    if s in "12345" and s:
                        scores[ax] = int(s); break
                    print("      ? 1-5, u, s, q")
                if skip:
                    break
            if skip or len(scores) < len(AXES):
                continue
            u = input("    unsupported_claim? [y/N] ").strip().lower().startswith("y") or unsupported
            fh.write(json.dumps(dict(case_id=r["case_id"], system=r["system"],
                                     **scores, unsupported_claim=u,
                                     composite=round(sum(scores.values()) / len(AXES), 3))) + "\n")
            fh.flush()
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted, progress saved")
    finally:
        fh.close()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
