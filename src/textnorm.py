"""Normalisation for TWCS text. The dataset is real Twitter output, so it carries
handles, t.co links, agent initials, part-markers and live PII."""
import re

RE_URL       = re.compile(r"https?://\S+")
RE_HANDLE    = re.compile(r"@\w+")
RE_AGENT_SIG = re.compile(r"\s*[\^~\-]\s?[A-Z]{2,3}\b\s*$")      # "... ^SC", "... -KC"
RE_PART      = re.compile(r"\s*\(?\d\s?/\s?\d\)?\s*")            # "1/2", "(1/2)"
RE_WS        = re.compile(r"\s+")

# PII. Order matters: order-ids before phone numbers, else the digit runs collide.
RE_ORDER  = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")                 # Amazon order id
RE_EMAIL  = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
RE_PHONE  = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s.-]?){2,4}\d{2,4}(?!\w)")
RE_CARD   = re.compile(r"\b(?:\d[ -]?){13,16}\b")

# Scripts that mean "not English" outright. AmazonHelp replies in ~12 languages.
RE_NONLATIN = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ"
    r"　-ヿ㐀-䶿一-鿿가-힯]")

# Function words that are common in English and rare in the Latin-script languages
# AmazonHelp actually uses (es/pt/fr/de/it/nl).
EN_STOP = {"the","to","and","you","your","for","with","that","this","have","has",
           "not","are","was","were","will","would","can","could","please","our",
           "we","us","it","is","of","on","in","my","me","at","be","from","do",
           "did","been","about","order","they","there","what","when","why","how"}
# Strong non-English markers (accents plus high-frequency foreign function words).
NON_EN_HINT = re.compile(
    r"[áéíóúñãõçâêôàèùäöüßîï]|"
    r"\b(?:que|para|por|con|una|nao|não|está|estoy|pero|como|muy|gracias|"
    r"ich|nicht|und|das|ist|mein|meine|habe|danke|wurde|noch|auch|"
    r"je|les|des|une|pour|pas|merci|bonjour|avec|"
    r"il|di|che|sono|grazie|"
    r"het|een|niet|mijn)\b", re.I)


def clean(text: str, keep_handles: bool = False) -> str:
    """Strip Twitter furniture and PII. Returns display-safe text."""
    if not isinstance(text, str):
        return ""
    t = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    t = RE_URL.sub("<link>", t)
    t = RE_ORDER.sub("<order-id>", t)
    t = RE_EMAIL.sub("<email>", t)
    t = RE_CARD.sub("<number>", t)
    t = RE_PHONE.sub("<phone>", t)
    if not keep_handles:
        t = RE_HANDLE.sub("", t)
    t = RE_AGENT_SIG.sub("", t)
    t = RE_PART.sub(" ", t)
    return RE_WS.sub(" ", t).strip()


def is_english(text: str) -> bool:
    """Cheap language gate. ponytail: heuristic, not a classifier — swap in
    fasttext-langid if the 3% error rate measured in notes/lang_audit.md matters."""
    if not text or len(text) < 12:
        return False
    if RE_NONLATIN.search(text):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return False
    hits = sum(w in EN_STOP for w in words)
    if NON_EN_HINT.search(text) and hits < 4:
        return False
    return hits >= 2


def demo():
    assert clean("@115850 order 406-0630403-8542757 call +919717981721 ^SC") == \
        "order <order-id> call <phone>"
    assert clean("Tips here: https://t.co/abc 1/2^AG") == "Tips here: <link>"
    assert clean("mail me at a.b+x@foo.co.uk please") == "mail me at <email> please"
    assert clean("fish &amp; chips") == "fish & chips"
    assert is_english("Where is my order, it has not arrived and I need it today")
    assert not is_english("Amazonプライム切ってなくて年会費取られてた")
    assert not is_english("Está bien pero si yo devuelvo el producto que compro otro")
    assert not is_english("Ich habe meine Bestellung nicht erhalten und das ist nicht gut")
    assert not is_english("ok")            # too short to judge
    print("textnorm ok")


if __name__ == "__main__":
    demo()
