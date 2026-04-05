"""Config templates for text."""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .....config import ConfigTemplate

try:
    import torch
except ImportError:
    torch = None

"""Generated config for CountVectorizer."""
class CountVectorizerConfig(ConfigTemplate):
    model_name = "CountVectorizer"
    model_path = "Code.models.machine_learning.feature_extraction.text.tex"

    def __init__(self,
        immutable: bool = True,
        input_files: Literal['filename', 'file', 'content'] = 'content',
        encoding: str = 'utf-8',
        decode_error: Literal['strict', 'ignore', 'replace'] = 'strict',
        strip_accents: Union[Literal['ascii', 'unicode'], Callable] = None,
        lowercase: bool = True,
        preprocessor: Callable = None,
        tokenizer: Union[Callable, object] = None,
        stop_words: Union[Literal['english'], list, tuple] = None,
        token_pattern: str = '(?u)\\b\\w\\w+\\b',
        ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
        analyzer: Union[Literal['word', 'char', 'char_wb'], Callable, object] = 'word',
        max_df: Union[int, float] = 1,
        min_df: Union[int, float] = 1,
        max_features: int = None,
        vocabulary: Union[Iterable, dict] = None,
        binary: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.input_files = input_files
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.device = device
        self.dtype = dtype


"""Generated config for FeatureHasher."""
class FeatureHasherConfig(ConfigTemplate):
    model_name = "FeatureHasher"
    model_path = "Code.models.machine_learning.feature_extraction.text.tex"

    def __init__(self,
        immutable: bool = True,
        n_features: int = 2 ** 20,
        input_type: str = 'dict',
        alternate_sign: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.n_features = n_features
        self.input_type = input_type
        self.alternate_sign = alternate_sign
        self.device = device
        self.dtype = dtype


"""Generated config for HashingVectorizer."""
class HashingVectorizerConfig(ConfigTemplate):
    model_name = "HashingVectorizer"
    model_path = "Code.models.machine_learning.feature_extraction.text.tex"

    def __init__(self,
        immutable: bool = True,
        input_files: Literal['filename', 'file', 'content'] = 'content',
        encoding: str = 'utf-8',
        decode_error: Literal['strict', 'ignore', 'replace'] = 'strict',
        strip_accents: Union[Literal['ascii', 'unicode'], Callable] = None,
        lowercase: bool = True,
        preprocessor: Callable = None,
        tokenizer: Union[Callable, object] = None,
        stop_words: Union[Literal['english'], list, tuple] = None,
        token_pattern: str = '(?u)\\b\\w\\w+\\b',
        ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
        analyzer: Union[Literal['word', 'char', 'char_wb'], Callable, object] = 'word',
        n_features: int = 1048576,
        binary: bool = False,
        norm: Union[Literal['l1', 'l2'], float, int] = 'l2',
        alternate_sign: bool = True,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.input_files = input_files
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.n_features = n_features
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.device = device
        self.dtype = dtype


"""Generated config for TfidfTransformer."""
class TfidfTransformerConfig(ConfigTemplate):
    model_name = "TfidfTransformer"
    model_path = "Code.models.machine_learning.feature_extraction.text.tex"

    def __init__(self,
        immutable: bool = True,
        norm: Union[Literal['l1', 'l2'], float, int] = 'l2',
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.device = device
        self.dtype = dtype


"""Generated config for TfidfVectorizer."""
class TfidfVectorizerConfig(ConfigTemplate):
    model_name = "TfidfVectorizer"
    model_path = "Code.models.machine_learning.feature_extraction.text.tex"

    def __init__(self,
        immutable: bool = True,
        input_files: Literal['filename', 'file', 'content'] = 'content',
        encoding: str = 'utf-8',
        decode_error: Literal['strict', 'ignore', 'replace'] = 'strict',
        strip_accents: Union[Literal['ascii', 'unicode'], Callable] = None,
        lowercase: bool = True,
        preprocessor: Callable = None,
        tokenizer: Union[Callable, object] = None,
        analyzer: Union[Literal['word', 'char', 'char_wb'], Callable, object] = 'word',
        stop_words: Union[Literal['english'], list, tuple] = None,
        token_pattern: str = '(?u)\\b\\w\\w+\\b',
        ngram_range: Union[list, tuple, torch.Tensor] = (1, 1),
        max_df: Union[int, float] = 1,
        min_df: Union[int, float] = 1,
        max_features: int = None,
        vocabulary: Union[Iterable, dict] = None,
        binary: bool = False,
        norm: Union[Literal['l1', 'l2'], float, int] = 'l2',
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float,
        **kwargs):
        super().__init__(immutable=immutable, **kwargs)
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
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.device = device
        self.dtype = dtype
