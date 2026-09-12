# Language-filter audit (`src/textnorm.py::is_english`)

AmazonHelp answers in ~12 languages; 30% of its reply pairs are non-English.
A classifier dependency was not worth adding, so the gate is a 15-line heuristic
(non-Latin script reject + English function-word count + foreign function-word hints).
Because a heuristic that silently mangles the corpus would invalidate every
downstream number, I audited it by hand instead of trusting it.

## Method
`scripts/../lang_audit` sampled 4,000 AmazonHelp reply pairs, split them by the
gate's verdict, and I read the first 45 of each side and labelled them myself.

## Result (n=90 hand-read)

| side | n read | my label disagrees | rate |
|---|---|---|---|
| kept as English  | 45 | 0  | **0%** (precision 100%) |
| dropped as non-English | 45 | 9 | **20%** (recall loss) |

Every one of the 9 misses is a short, context-dependent fragment:
`"I needed it today..."`, `"Still not here*"`, `"I got charged....."`,
`"fixed. thanks! :)"`, `"See there's no reply"`, `"My contact no <phone>"`,
`"It's dance quiz."`, and two off-topic quips.

## Why this trade is the right one here
The gate is **precision-oriented**: nothing foreign leaks into the corpus the
agent grounds its replies in, which is what would actually corrupt output. The
cost is ~20% of 51,661 drops (~10k) short English fragments lost. Those
fragments are unusable as standalone support messages anyway — they carry no
resolvable request without their parent thread.

**Consequence to carry into the report:** the corpus is biased toward *longer,
self-contained* customer messages. Terse real traffic ("still not here") is
under-represented, so measured performance is optimistic for that slice.
