"""Extract (customer message -> brand reply) cases for one brand from twcs.csv.

Produces a TEMPORAL split so retrieval can never see the future:
  data/processed/corpus.jsonl  - earlier window, the agent's knowledge base
  data/processed/eval_pool.jsonl - later window, what golden examples are drawn from
"""
import argparse, json, sys, pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from textnorm import clean, is_english

OUT = pathlib.Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--split-quantile", type=float, default=0.70,
                    help="fraction of the timeline that becomes the retrieval corpus")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    df["created_at"] = pd.to_datetime(df.created_at, format="%a %b %d %H:%M:%S %z %Y",
                                      errors="coerce", utc=True)
    byid = df.set_index("tweet_id")

    replies = df[(df.author_id == args.brand) & (~df.inbound) &
                 df.in_response_to_tweet_id.notna()]
    print(f"{args.brand}: {len(replies)} brand replies", file=sys.stderr)

    rows, dropped = [], {"no_parent": 0, "parent_outbound": 0, "not_english": 0, "too_short": 0}
    for r in replies.itertuples():
        try:
            cust = byid.loc[int(r.in_response_to_tweet_id)]
        except KeyError:
            dropped["no_parent"] += 1
            continue
        if isinstance(cust, pd.DataFrame):      # duplicate tweet_id in source data
            cust = cust.iloc[0]
        if not cust.inbound:
            dropped["parent_outbound"] += 1
            continue

        msg, reply = clean(cust.text), clean(r.text)
        if not is_english(msg):
            dropped["not_english"] += 1
            continue
        if len(msg) < 15 or len(reply) < 15:
            dropped["too_short"] += 1
            continue

        # one turn of prior context, when the thread has one
        ctx = ""
        if pd.notna(cust.in_response_to_tweet_id):
            try:
                p = byid.loc[int(cust.in_response_to_tweet_id)]
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[0]
                ctx = clean(p.text)
            except KeyError:
                pass

        rows.append(dict(case_id=f"{args.brand}-{int(r.tweet_id)}",
                         customer_msg=msg, brand_reply=reply, prior_turn=ctx,
                         created_at=cust.created_at if pd.notna(cust.created_at) else r.created_at,
                         customer_id=str(cust.author_id)))

    out = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    # A customer can appear in several cases; keep each customer wholly on one
    # side of the split so a golden example's own thread is never retrievable.
    cut = out.created_at.quantile(args.split_quantile)
    late_customers = set(out.loc[out.created_at > cut, "customer_id"])
    is_eval = out.customer_id.isin(late_customers)

    OUT.mkdir(parents=True, exist_ok=True)
    corpus, pool = out[~is_eval], out[is_eval]
    for name, part in (("corpus", corpus), ("eval_pool", pool)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for rec in part.to_dict("records"):
                rec["created_at"] = rec["created_at"].isoformat()
                f.write(json.dumps(rec) + "\n")

    print(f"dropped: {dropped}", file=sys.stderr)
    print(f"corpus={len(corpus)} (<= {cut})  eval_pool={len(pool)} (> {cut})", file=sys.stderr)
    assert not (set(corpus.customer_id) & set(pool.customer_id)), "customer leaked across split"
    (OUT / "split.json").write_text(json.dumps(dict(
        brand=args.brand, split_at=str(cut), n_corpus=len(corpus),
        n_eval_pool=len(pool), dropped=dropped), indent=2))


if __name__ == "__main__":
    main()
