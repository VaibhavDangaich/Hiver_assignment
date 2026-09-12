"""Human adjudication CLI. This is the only script that writes gold labels.

Why it exists: two machine passes produce *candidate* labels. A gold set whose
labels were never seen by a human is not a gold set, and every number computed
against it would be "model agrees with model". This tool puts a human on every
item that matters and records, per item, whether the human accepted or changed
the machine proposal -- so the report can state exactly how much human judgement
is in the reference.

Queue order (most decision-relevant first):
  1. A/B disagreed on intent or action   -- genuinely contested
  2. flagged ambiguous by pass A          -- self-reported doubt
  3. everything else                      -- confirm-by-default, still shown

Usage:
  python scripts/adjudicate.py            # full queue
  python scripts/adjudicate.py --only-contested
  python scripts/adjudicate.py --resume
Keys: Enter accept | number pick intent | a/e force action | s skip | q save+quit
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from taxonomy import INTENTS, ESCALATION_RULES, default_action

GOLD = pathlib.Path("data/golden/labelled.jsonl")
KEYS = list(INTENTS)


def load_done() -> dict:
    if not GOLD.exists():
        return {}
    out = {}
    for line in GOLD.open():
        try:
            r = json.loads(line)
            out[r["case_id"]] = r
        except json.JSONDecodeError:
            continue
    return out


def priority(r) -> int:
    a, b = r["pre"], r.get("pre_b", {})
    if a["intent"] != b.get("intent") or a["action"] != b.get("action"):
        return 0
    return 1 if a.get("ambiguous") else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-contested", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data/golden/preannotated.jsonl")]
    done = load_done() if args.resume else {}
    queue = sorted((r for r in rows if r["case_id"] not in done), key=priority)
    if args.only_contested:
        queue = [r for r in queue if priority(r) == 0]

    print(f"\n{len(queue)} to adjudicate ({len(done)} already done)")
    print("contested:", sum(1 for r in queue if priority(r) == 0),
          "| ambiguous:", sum(1 for r in queue if priority(r) == 1),
          "| routine:", sum(1 for r in queue if priority(r) == 2))
    print("\nINTENTS:")
    for i, k in enumerate(KEYS):
        print(f"  {i:2d} {k}")
    print("\nEnter=accept  <n>=intent  a/e=action  s=skip  q=save+quit\n" + "=" * 78)

    fh = GOLD.open("a")
    try:
        for n, r in enumerate(queue, 1):
            a, b = r["pre"], r.get("pre_b", {})
            tag = ["CONTESTED", "ambiguous", "routine"][priority(r)]
            print(f"\n[{n}/{len(queue)}] {r['case_id']}  ({r['slice']}, {tag})")
            if r.get("prior_turn"):
                print(f"  prior: {r['prior_turn'][:160]}")
            print(f"  MSG: {r['customer_msg']}")
            print(f"  brand actually replied: {r['brand_reply'][:170]}")
            print(f"  A: {a['intent']:24s} {a['action']:8s} {a.get('reason','')[:70]}")
            print(f"  B: {str(b.get('intent')):24s} {str(b.get('action')):8s} {str(b.get('reason',''))[:70]}")
            if a.get("notes"):
                print(f"  note: {a['notes'][:150]}")

            intent, action = a["intent"], a["action"]
            changed = False
            while True:
                s = input(f"  -> [{intent} / {action}] ").strip().lower()
                if s == "":
                    break
                if s == "q":
                    print(f"saved {n - 1} this session"); fh.close(); return
                if s == "s":
                    intent = None; break
                if s in ("a", "auto"):
                    action, changed = "auto", True; continue
                if s in ("e", "esc", "escalate"):
                    action, changed = "escalate", True; continue
                if s.isdigit() and int(s) < len(KEYS):
                    intent, changed = KEYS[int(s)], True
                    action = default_action(intent)
                    print(f"     intent={intent}, action defaulted to {action}")
                    continue
                print("     ? Enter/number/a/e/s/q")
            if intent is None:
                continue
            fh.write(json.dumps(dict(
                case_id=r["case_id"], slice=r["slice"], customer_msg=r["customer_msg"],
                prior_turn=r.get("prior_turn", ""), brand_reply=r["brand_reply"],
                gold_intent=intent, gold_action=action,
                human_changed=changed, queue_tag=tag,
                pre_a=dict(intent=a["intent"], action=a["action"]),
                pre_b=dict(intent=b.get("intent"), action=b.get("action")),
            )) + "\n")
            fh.flush()
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted, progress saved")
    finally:
        fh.close()
    print(f"\nwrote {GOLD}")


if __name__ == "__main__":
    main()
