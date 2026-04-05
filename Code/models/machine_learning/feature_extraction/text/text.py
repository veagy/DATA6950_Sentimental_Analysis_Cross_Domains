import re
import struct
import unicodedata
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal, Iterable
from .....models.utils import MLModule
import numpy as np
from torch.func import vmap
import joblib


__all__ = [
    "HashingVectorizer",
    "CountVectorizer",
    "TfidfVectorizer",
    "FeatureHasher",
    "TfidfTransformer",
]

# ---------------------------------------------------------------------------
# English stop words (sklearn-compatible set)
# ---------------------------------------------------------------------------
ENGLISH_STOP_WORDS: frozenset = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "will", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
])

# ---------------------------------------------------------------------------
# Murmurhash3 (32-bit signed) — deterministic hashing trick
# ---------------------------------------------------------------------------

def _murmurhash3_32(data: bytes, seed: int = 0) -> int:
    """32-bit signed Murmurhash3 — matches sklearn's mmh3 behaviour."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    length = len(data)
    h = seed & 0xFFFFFFFF
    n_blocks = length >> 2
    for i in range(n_blocks):
        k = struct.unpack_from("<I", data, i << 2)[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = ((h * 5) & 0xFFFFFFFF + 0xE6546B64) & 0xFFFFFFFF
    tail = data[n_blocks << 2:]
    k = 0
    tlen = len(tail)
    if tlen >= 3:
        k ^= tail[2] << 16
    if tlen >= 2:
        k ^= tail[1] << 8
    if tlen >= 1:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    # finalisation
    h ^= length
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return int(h) - 0x100000000 if h >= 0x80000000 else int(h)


# ---------------------------------------------------------------------------
# Shared text-processing helpers
# ---------------------------------------------------------------------------

def _load_document(doc: Any, input_files: str, encoding: str, decode_error: str) -> str:
    """Read raw text from a document depending on input_files mode."""
    if input_files == "filename":
        with open(doc, "rb") as f:
            raw = f.read()
        return raw.decode(encoding, errors=decode_error)
    if input_files == "file":
        raw = doc.read()
        if isinstance(raw, bytes):
            return raw.decode(encoding, errors=decode_error)
        return raw
    # "content"
    if isinstance(doc, bytes):
        return doc.decode(encoding, errors=decode_error)
    return str(doc)


def _strip_accents_ascii(text: str) -> str:
    """Strip accents using ASCII normalization (fast, ASCII chars only)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if ord(c) < 128)


def _strip_accents_unicode(text: str) -> str:
    """Strip accents using full Unicode NFKD normalization."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _preprocess(
    text: str,
    strip_accents: Union[str, Callable, None],
    lowercase: bool,
    preprocessor: Optional[Callable],
) -> str:
    """Apply lowercasing and accent stripping (or custom preprocessor)."""
    if preprocessor is not None:
        return preprocessor(text)
    if lowercase:
        text = text.lower()
    if strip_accents == "ascii":
        text = _strip_accents_ascii(text)
    elif strip_accents == "unicode":
        text = _strip_accents_unicode(text)
    elif callable(strip_accents):
        text = strip_accents(text)
    return text


def _make_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    """Expand a token list into n-grams."""
    min_n, max_n = ngram_range
    if min_n == max_n == 1:
        return tokens
    result: List[str] = []
    n_tokens = len(tokens)
    for n in range(min_n, min(max_n, n_tokens) + 1):
        for i in range(n_tokens - n + 1):
            result.append(" ".join(tokens[i: i + n]))
    return result


def _make_char_ngrams(text: str, ngram_range: Tuple[int, int]) -> List[str]:
    """Character n-grams from raw text."""
    min_n, max_n = ngram_range
    result: List[str] = []
    n_chars = len(text)
    for n in range(min_n, min(max_n, n_chars) + 1):
        for i in range(n_chars - n + 1):
            result.append(text[i: i + n])
    return result


def _make_char_wb_ngrams(text: str, ngram_range: Tuple[int, int]) -> List[str]:
    """Character n-grams only from within word boundaries (padded with space)."""
    result: List[str] = []
    for word in text.split():
        padded = f" {word} "
        result.extend(_make_char_ngrams(padded, ngram_range))
    return result


def _resolve_stop_words(stop_words: Any) -> Optional[frozenset]:
    """Resolve stop_words argument to a frozenset or None."""
    if stop_words is None:
        return None
    if isinstance(stop_words, str) and stop_words.lower() == "english":
        return ENGLISH_STOP_WORDS
    if isinstance(stop_words, (list, tuple, set, frozenset)):
        return frozenset(stop_words)
    return None


def _build_analyzer(
    input_files: str,
    encoding: str,
    decode_error: str,
    strip_accents: Any,
    lowercase: bool,
    preprocessor: Optional[Callable],
    tokenizer: Optional[Callable],
    stop_words_set: Optional[frozenset],
    token_pattern: Optional[str],
    ngram_range: Tuple[int, int],
    analyzer: Any,
) -> Callable[[Any], List[str]]:
    """
    Build and return an analyzer callable (raw_doc -> token list).
    Handles 'word', 'char', 'char_wb', or a custom callable.
    """
    if callable(analyzer):
        def _callable_analyzer(doc):
            raw = _load_document(doc, input_files, encoding, decode_error)
            return list(analyzer(raw))
        return _callable_analyzer

    def _analyze(doc):
        raw = _load_document(doc, input_files, encoding, decode_error)
        text = _preprocess(raw, strip_accents, lowercase, preprocessor)

        if analyzer == "char":
            return _make_char_ngrams(text, ngram_range)

        if analyzer == "char_wb":
            return _make_char_wb_ngrams(text, ngram_range)

        # analyzer == "word" (default)
        if tokenizer is not None:
            tokens = list(tokenizer(text))
        elif token_pattern is not None:
            pat = re.compile(token_pattern)
            groups = pat.groupindex
            if groups:
                tokens = pat.findall(text)
            else:
                # Use capturing group if present, else full match
                tokens = pat.findall(text) if pat.groups else pat.findall(text)
        else:
            tokens = re.findall(r"(?u)\b\w\w+\b", text)

        if stop_words_set:
            tokens = [t for t in tokens if t not in stop_words_set]

        tokens = _make_ngrams(tokens, ngram_range)
        return tokens

    return _analyze


def _normalize_rows(X: torch.Tensor, norm: Optional[str]) -> torch.Tensor:
    """L1 or L2 row-normalization of a dense tensor."""
    if norm is None:
        return X
    if norm == "l2":
        norms = torch.norm(X, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return X / norms
    if norm == "l1":
        norms = torch.norm(X, p=1, dim=1, keepdim=True).clamp(min=1e-12)
        return X / norms
    if isinstance(norm, (int, float)):
        norms = torch.norm(X, p=float(norm), dim=1, keepdim=True).clamp(min=1e-12)
        return X / norms
    return X


# ---------------------------------------------------------------------------
# HashingVectorizer
# ---------------------------------------------------------------------------

class HashingVectorizer(MLModule):
    def __init__(self,
                 input_files: Literal["filename", "file", "content"] = 'content',
                 encoding: str = 'utf-8',
                 decode_error: Literal["strict", "ignore", "replace"] = 'strict',
                 strip_accents: Union[Literal["ascii", "unicode"], Callable] = None,
                 lowercase: bool = True,
                 preprocessor: Callable = None,
                 tokenizer: Union[Callable, object] = None,
                 stop_words: Union[Literal["english"], list, tuple] = None,
                 token_pattern: str = r'(?u)\b\w\w+\b',
                 ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
                 analyzer: Union[Literal["word", "char", "char_wb"], Callable, object] = 'word',
                 n_features: int = 1048576,
                 binary: bool = False,
                 norm: Union[Literal["l1", "l2"], float, int] = 'l2',
                 alternate_sign: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.input_files = input_files
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = tuple(ngram_range) if not isinstance(ngram_range, tuple) else ngram_range
        self.analyzer = analyzer
        self.n_features = n_features
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # HashingVectorizer is stateless (no learned vocabulary)
        self.fit_status = True

    def _get_analyzer(self) -> Callable:
        stop_words_set = _resolve_stop_words(self.stop_words)
        return _build_analyzer(
            input_files=self.input_files,
            encoding=self.encoding,
            decode_error=self.decode_error,
            strip_accents=self.strip_accents,
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            tokenizer=self.tokenizer if callable(self.tokenizer) else None,
            stop_words_set=stop_words_set,
            token_pattern=self.token_pattern,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
        )

    def _hash_token(self, token: str) -> Tuple[int, int]:
        """Hash a token → (feature_index, sign)."""
        h = _murmurhash3_32(token.encode("utf-8"))
        idx = abs(h) % self.n_features
        sign = 1 if (not self.alternate_sign or h >= 0) else -1
        return idx, sign

    def _doc_to_vector(self, tokens: List[str]) -> torch.Tensor:
        """Convert token list to a feature vector of length n_features."""
        vec = torch.zeros(self.n_features, device=self.device, dtype=self.dtype)
        for token in tokens:
            idx, sign = self._hash_token(token)
            vec[idx] += sign
        if self.binary:
            vec = (vec != 0).to(self.dtype)
        return vec

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "HashingVectorizer":
        """No-op — HashingVectorizer is stateless."""
        self.fit_status = True
        return self

    def transform(
        self,
        X: Iterable,
        **kwargs,
    ) -> torch.Tensor:
        analyze = self._get_analyzer()
        docs = list(X)
        n = len(docs)
        result = torch.zeros(n, self.n_features, device=self.device, dtype=self.dtype)
        for i, doc in enumerate(docs):
            tokens = analyze(doc)
            result[i] = self._doc_to_vector(tokens)

        result = _normalize_rows(result, self.norm)
        return result

    def fit_transform(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit (no-op) and transform X."""
        return self.transform(X, **kwargs)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# CountVectorizer
# ---------------------------------------------------------------------------

class CountVectorizer(MLModule):
    def __init__(self,
                 input_files: Literal["filename", "file", "content"] = 'content',
                 encoding: str = 'utf-8',
                 decode_error: Literal["strict", "ignore", "replace"] = 'strict',
                 strip_accents: Union[Literal["ascii", "unicode"], Callable] = None,
                 lowercase: bool = True,
                 preprocessor: Callable = None,
                 tokenizer: Union[Callable, object] = None,
                 stop_words: Union[Literal["english"], list, tuple] = None,
                 token_pattern: str = r'(?u)\b\w\w+\b',
                 ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
                 analyzer: Union[Literal["word", "char", "char_wb"], Callable, object] = 'word',
                 max_df: Union[int, float] = 1,
                 min_df: Union[int, float] = 1,
                 max_features: int = None,
                 vocabulary: Union[Iterable, dict] = None,
                 binary: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.input_files = input_files
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = tuple(ngram_range) if not isinstance(ngram_range, tuple) else ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.vocabulary_: Dict[str, int] = {}
        self.fixed_vocabulary_: bool = False
        self.stop_words_: Optional[frozenset] = None
        self.fit_status = False

    def _get_analyzer(self) -> Callable:
        stop_words_set = _resolve_stop_words(self.stop_words)
        self.stop_words_ = stop_words_set
        return _build_analyzer(
            input_files=self.input_files,
            encoding=self.encoding,
            decode_error=self.decode_error,
            strip_accents=self.strip_accents,
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            tokenizer=self.tokenizer if callable(self.tokenizer) else None,
            stop_words_set=stop_words_set,
            token_pattern=self.token_pattern,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
        )

    def _limit_features(
        self,
        term_doc_freqs: Dict[str, int],
        term_total_freqs: Dict[str, int],
        n_docs: int,
    ) -> Dict[str, int]:
        """Apply min_df, max_df, max_features filters and return final vocab."""
        max_df = self.max_df
        min_df = self.min_df

        if isinstance(max_df, float):
            max_doc_count = int(max_df * n_docs)
        else:
            max_doc_count = int(max_df)

        if isinstance(min_df, float):
            min_doc_count = int(min_df * n_docs)
        else:
            min_doc_count = int(min_df)
        min_doc_count = max(min_doc_count, 1)

        filtered = {
            term: freq
            for term, freq in term_doc_freqs.items()
            if min_doc_count <= freq <= max_doc_count
        }

        if self.max_features is not None:
            # Keep top-max_features by total corpus frequency
            sorted_terms = sorted(
                filtered.keys(),
                key=lambda t: -term_total_freqs.get(t, 0),
            )
            sorted_terms = sorted_terms[: self.max_features]
            filtered = {t: filtered[t] for t in sorted_terms}

        # Sort alphabetically for deterministic output
        vocab = {term: idx for idx, term in enumerate(sorted(filtered.keys()))}
        return vocab

    def fit(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "CountVectorizer":
        # Fixed vocabulary provided by user
        if self.vocabulary is not None:
            if isinstance(self.vocabulary, dict):
                self.vocabulary_ = dict(self.vocabulary)
            else:
                self.vocabulary_ = {term: idx for idx, term in enumerate(self.vocabulary)}
            self.fixed_vocabulary_ = True
            self.fit_status = True
            return self

        self.fixed_vocabulary_ = False
        analyze = self._get_analyzer()
        docs = list(X)
        n_docs = len(docs)

        term_doc_freqs: Dict[str, int] = {}
        term_total_freqs: Dict[str, int] = {}

        for doc in docs:
            tokens = analyze(doc)
            seen = set()
            for token in tokens:
                term_total_freqs[token] = term_total_freqs.get(token, 0) + 1
                if token not in seen:
                    term_doc_freqs[token] = term_doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self.vocabulary_ = self._limit_features(term_doc_freqs, term_total_freqs, n_docs)
        self.fit_status = True
        return self

    def transform(
        self,
        X: Iterable,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("CountVectorizer is not fitted. Call fit() first.")

        analyze = self._get_analyzer()
        docs = list(X)
        n = len(docs)
        n_features = len(self.vocabulary_)
        result = torch.zeros(n, n_features, device=self.device, dtype=self.dtype)

        for i, doc in enumerate(docs):
            tokens = analyze(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    result[i, self.vocabulary_[token]] += 1

        if self.binary:
            result = (result > 0).to(self.dtype)
        return result

    def fit_transform(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit the model and transform X."""
        return self.fit(X, y, **kwargs).transform(X)

    def inverse_transform(self, X: Any) -> List[List[str]]:
        if not self.fit_status:
            raise RuntimeError("CountVectorizer is not fitted. Call fit() first.")
        if isinstance(X, torch.Tensor) and X.is_sparse:
            X = X.to_dense()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        idx_to_term = {v: k for k, v in self.vocabulary_.items()}
        result = []
        for row in X:
            terms = [idx_to_term[j] for j in range(len(row)) if row[j].item() != 0]
            result.append(terms)
        return result

    def get_feature_names_out(self) -> List[str]:
        """Return sorted feature names."""
        if not self.fit_status:
            raise RuntimeError("CountVectorizer is not fitted. Call fit() first.")
        return sorted(self.vocabulary_.keys(), key=lambda t: self.vocabulary_[t])

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# TfidfTransformer
# ---------------------------------------------------------------------------

class TfidfTransformer(MLModule):
    def __init__(self,
                 norm: Union[Literal["l1", "l2"], float, int] = 'l2',
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Fitted attributes
        self.idf_: Optional[torch.Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.fit_status = False

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "TfidfTransformer":
        if isinstance(X, torch.Tensor) and X.is_sparse:
            X = X.to_dense()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=self.device, dtype=torch.float64)
        else:
            X = X.to(device=self.device, dtype=torch.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.use_idf:
            df = (X > 0).sum(dim=0).float()
            if self.smooth_idf:
                idf = torch.log((1.0 + n_samples) / (1.0 + df)) + 1.0
            else:
                idf = torch.log(n_samples / df.clamp(min=1.0)) + 1.0
            self.idf_ = idf.to(dtype=self.dtype, device=self.device)
        else:
            self.idf_ = torch.ones(n_features, device=self.device, dtype=self.dtype)

        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        copy: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("TfidfTransformer is not fitted. Call fit() first.")

        if isinstance(X, torch.Tensor) and X.is_sparse:
            X = X.to_dense()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        else:
            if copy:
                X = X.clone()
            X = X.to(device=self.device, dtype=self.dtype)

        if self.sublinear_tf:
            # 1 + log(tf), but log(0) = -inf → use clamp
            X = torch.where(X > 0, 1.0 + torch.log(X.clamp(min=1.0)), torch.zeros_like(X))

        if self.use_idf and self.idf_ is not None:
            X = X * self.idf_.unsqueeze(0)

        X = _normalize_rows(X, self.norm)
        return X

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# TfidfVectorizer
# ---------------------------------------------------------------------------

class TfidfVectorizer(MLModule):
    def __init__(self,
                 input_files: Literal["filename", "file", "content"] = 'content',
                 encoding: str = 'utf-8',
                 decode_error: Literal["strict", "ignore", "replace"] = 'strict',
                 strip_accents: Union[Literal["ascii", "unicode"], Callable] = None,
                 lowercase: bool = True,
                 preprocessor: Callable = None,
                 tokenizer: Union[Callable, object] = None,
                 analyzer: Union[Literal["word", "char", "char_wb"], Callable, object] = 'word',
                 stop_words: Union[Literal["english"], list, tuple] = None,
                 token_pattern: str = r'(?u)\b\w\w+\b',
                 ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
                 max_df: Union[int, float] = 1,
                 min_df: Union[int, float] = 1,
                 max_features: int = None,
                 vocabulary: Union[Iterable, dict] = None,
                 binary: bool = False,
                 norm: Union[Literal["l1", "l2"], float, int] = 'l2',
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.input_files = input_files
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = tuple(ngram_range) if not isinstance(ngram_range, tuple) else ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        # Internal sub-transformers
        self._cv = CountVectorizer(
            input_files=input_files, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor,
            tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=self.ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            device=device, dtype=dtype,
        )
        self._tfidf = TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf, device=device, dtype=dtype,
        )
        # Fitted attributes (proxied from sub-transformers)
        self.vocabulary_: Dict[str, int] = {}
        self.fixed_vocabulary_: bool = False
        self.idf_: Optional[torch.Tensor] = None
        self.fit_status = False

    def fit(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "TfidfVectorizer":
        counts = self._cv.fit_transform(X, y)
        self._tfidf.fit(counts)
        self.vocabulary_ = self._cv.vocabulary_
        self.fixed_vocabulary_ = self._cv.fixed_vocabulary_
        self.idf_ = self._tfidf.idf_
        self.fit_status = True
        return self

    def transform(
        self,
        X: Iterable,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            raise RuntimeError("TfidfVectorizer is not fitted. Call fit() first.")
        counts = self._cv.transform(X)
        return self._tfidf.transform(counts)

    def fit_transform(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit the model and transform X."""
        counts = self._cv.fit_transform(X, y)
        tfidf = self._tfidf.fit_transform(counts)
        self.vocabulary_ = self._cv.vocabulary_
        self.fixed_vocabulary_ = self._cv.fixed_vocabulary_
        self.idf_ = self._tfidf.idf_
        self.fit_status = True
        return tfidf

    def inverse_transform(self, X: Any) -> List[List[str]]:
        """Return terms per document with nonzero entries in X."""
        if not self.fit_status:
            raise RuntimeError("TfidfVectorizer is not fitted. Call fit() first.")
        return self._cv.inverse_transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return sorted feature names."""
        if not self.fit_status:
            raise RuntimeError("TfidfVectorizer is not fitted. Call fit() first.")
        return self._cv.get_feature_names_out()

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fit_status:
            return self.fit_transform(X, y, **kwargs)
        return self.transform(X, **kwargs)


# ---------------------------------------------------------------------------
# FeatureHasher
# ---------------------------------------------------------------------------

class FeatureHasher(MLModule):
    def __init__(self,
                 n_features: int = 2 ** 20,
                 input_type: str = "dict",
                 alternate_sign: bool = True,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.n_features = n_features
        self.input_type = input_type
        self.alternate_sign = alternate_sign
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # Stateless transformer
        self.fit_status = True

    def _hash_feature(self, name: str) -> Tuple[int, int]:
        """Hash a feature name → (column_index, sign)."""
        if isinstance(name, bytes):
            raw = name
        else:
            raw = name.encode("utf-8")
        h = _murmurhash3_32(raw)
        idx = abs(h) % self.n_features
        sign = 1 if (not self.alternate_sign or h >= 0) else -1
        return idx, sign

    def _extract_pairs(self, sample: Any) -> List[Tuple[str, float]]:
        if self.input_type == "dict":
            if isinstance(sample, dict):
                return [(str(k), float(v)) for k, v in sample.items()]
            raise ValueError(f"Expected dict, got {type(sample)}")

        if self.input_type == "pair":
            # Sample is an iterable of (name, value) pairs
            return [(str(k), float(v)) for k, v in sample]

        if self.input_type == "string":
            # Sample is an iterable of strings; each implies value=1
            if isinstance(sample, str):
                return [(sample, 1.0)]
            return [(str(s), 1.0) for s in sample]

        raise ValueError(
            f"input_type must be 'dict', 'pair', or 'string'; got '{self.input_type}'"
        )

    def fit(
        self,
        X: Any = None,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "FeatureHasher":
        """No-op — FeatureHasher is stateless."""
        self.fit_status = True
        return self

    def transform(
        self,
        X: Iterable,
        **kwargs,
    ) -> torch.Tensor:
        samples = list(X)
        n = len(samples)
        result = torch.zeros(n, self.n_features, device=self.device, dtype=self.dtype)

        for i, sample in enumerate(samples):
            pairs = self._extract_pairs(sample)
            for name, val in pairs:
                idx, sign = self._hash_feature(name)
                result[i, idx] += sign * val

        return result

    def fit_transform(
        self,
        X: Iterable,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit (no-op) and transform X."""
        return self.transform(X, **kwargs)

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.transform(X, **kwargs)
