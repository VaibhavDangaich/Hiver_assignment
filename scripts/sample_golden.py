"""Sample the golden evaluation set from the held-out (later) eval_pool.

Two slices, deliberately never merged into one headline number:

  random   (n=150) uniform draw -> reflects the real traffic mix, so it is the
           number that speaks to production behaviour.
  enriched (n=70)  keyword-stratified draw over intents that are rare but
           consequential (fraud, account lockout, billing, tech) -> a diagnostic
           slice with enough positives per class for per-class metrics to mean
           anything. Reported separately; averaging it in would flatter macro-F1.
"""
import argparse, json, pathlib, random, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# Transparent, weak keyword strata. These only steer the SAMPLE; they are never
# used as labels and never seen by the agent.
STRATA = {
    "fraud_report":   r"\b(scam|phish|fraud|fake|counterfeit|impersonat|spam email|someone (else )?used)\b",
    "account_access": r"\b(locked|suspend|can'?t (log|sign) ?in|cannot (log|sign) ?in|password|reset my|blocked my account|2fa)\b",
    "billing_charge": r"\b(charged|charge|cashback|gift card|balance|overcharg|double.?charg|amazon pay|wrong price|refund.{0,10}amount)\b",
    "tech_support":   r"\b(app|website|kindle|fire ?tv|prime video|playback|buffer|error|crash|won'?t load|not working)\b",
    "order_change":   r"\b(cancel|change (the )?address|change my address|modify|wrong address)\b",
    "return_replace": r"\b(return|replace|replacement|damaged|broken|faulty|defect|wrong item)\b",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--n-enriched", type=int, default=70)
    ap.add_argument("--seed", type=int, default=20240917)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open("data/processed/eval_pool.jsonl")]
    # Keep it human-readable and self-contained enough to label fairly.
    pool = [r for r in pool if 25 <= len(r["customer_msg"]) <= 400]
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    taken: set[str] = set()
    rand_slice = pool[:args.n_random]
    taken.update(r["case_id"] for r in rand_slice)

    rest = [r for r in pool if r["case_id"] not in taken]
    per = args.n_enriched // len(STRATA)
    enriched = []
    for name, pat in STRATA.items():
        rx = re.compile(pat, re.I)
        hits = [r for r in rest if rx.search(r["customer_msg"])
                and r["case_id"] not in taken]
        for r in hits[:per]:
            taken.add(r["case_id"])
            enriched.append({**r, "stratum": name})
    # top up to exactly n_enriched from whatever matched any stratum
    if len(enriched) < args.n_enriched:
        anyrx = re.compile("|".join(STRATA.values()), re.I)
        for r in rest:
            if len(enriched) >= args.n_enriched:
                break
            if r["case_id"] not in taken and anyrx.search(r["customer_msg"]):
                taken.add(r["case_id"])
                enriched.append({**r, "stratum": "mixed"})

    out = pathlib.Path("data/golden")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("random", rand_slice), ("enriched", enriched)):
        with open(out / f"unlabelled_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({**r, "slice": name}) + "\n")
        print(f"{name}: {len(rows)}", file=sys.stderr)

    ids = [r["case_id"] for r in rand_slice] + [r["case_id"] for r in enriched]
    assert len(ids) == len(set(ids)), "slices overlap"
    corpus_ids = {json.loads(l)["case_id"] for l in open("data/processed/corpus.jsonl")}
    assert not (set(ids) & corpus_ids), "golden example present in retrieval corpus"
    print(f"total {len(ids)}, no overlap with corpus", file=sys.stderr)


if __name__ == "__main__":
    main()
