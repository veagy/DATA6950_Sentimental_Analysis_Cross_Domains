import re
import html
import unicodedata
import pandas as pd


def clean_text(text: str, *, lowercase: bool = True) -> str:
    """
    Full text cleaning pipeline:
    1. Decode HTML entities (&amp; &lt; etc.)
    2. Strip HTML/XML tags
    3. Remove URLs (http, https, ftp, www)
    4. Remove email addresses
    5. Remove Twitter handles and hashtags
    6. Normalise Unicode (NFKC — merges compatibility characters)
    7. Remove non-printable / control characters
    8. Collapse repeated punctuation (!!! → !)
    9. Collapse whitespace
    10. Optionally lowercase
    """
    if not isinstance(text, str):
        return str(text) if pd.notna(text) else ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?:https?://|ftp://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)   # strip non-ASCII control chars
    text = re.sub(r"([!?.]){2,}", r"\1", text)        # !!! → !
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def clean_code(text: str) -> str:
    """Remove fenced and inline code blocks from mixed text+code content."""
    if not isinstance(text, str):
        return str(text) if pd.notna(text) else ""
        
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
