"""Fetch the TWCS dataset.

Primary source is the Kaggle dataset named in the assignment
(thoughtvector/customer-support-on-twitter). Kaggle requires credentials, so
this defaults to a public Hugging Face mirror of the same file and verifies the
size and header before use. Cite: see README "Data provenance".
"""
import hashlib, pathlib, sys, urllib.request

URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
DEST = pathlib.Path("data/raw/twcs.csv")
EXPECTED_BYTES = 516508641
EXPECTED_HEADER = ("tweet_id,author_id,inbound,created_at,text,"
                   "response_tweet_id,in_response_to_tweet_id")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size == EXPECTED_BYTES:
        print(f"{DEST} already present ({EXPECTED_BYTES} bytes)")
    else:
        print(f"downloading {URL} -> {DEST} (516MB)...")
        with urllib.request.urlopen(URL) as r, DEST.open("wb") as f:
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done/1e6:.0f} MB", end="", flush=True)
        print()
    size = DEST.stat().st_size
    header = DEST.open().readline().strip()
    if header != EXPECTED_HEADER:
        sys.exit(f"unexpected header:\n  got      {header}\n  expected {EXPECTED_HEADER}")
    if size != EXPECTED_BYTES:
        print(f"WARNING: size {size} != expected {EXPECTED_BYTES}; the mirror may have changed.")
    print(f"ok: {size} bytes, header verified")


if __name__ == "__main__":
    main()
