"""
Phase 4: Modality-Specific NLP Preprocessing.
Supports PyTorch SFT dynamic collation, ignore-index masking, and AutoTokenizers.
"""

import re
import unicodedata
from typing import List, Dict, Any, Union

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace is optional but fundamentally required for standard transformers.
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizerBase = Any

IGNORE_INDEX = -100

# -----------------------------------------------------------------------------
# 1. CLASSICAL NLP PIPELINE
# -----------------------------------------------------------------------------

def classical_nlp_pipeline(text: str, method: str = "stem") -> List[str]:
    """
    Classical NLP string cleaning pipeline.
    Strips HTML, normalises Unicode, removes non-alphanumeric chars.
    Applies stemming or basic tokenisation.
    (SpaCy/NLTK imports isolated to runtime to prevent massive local overhead).
    
    Args:
        text: Raw text string to process.
        method: "stem" (fast NLTK PorterStemmer) or "none".
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Clean text (Regex compilation is fast enough inline here)
    text = re.sub(r"<[^>]+>", " ", text)               # strip HTML
    text = re.sub(r"https?://\S+", " ", text)           # remove URLs
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(text)

    if method == "stem":
        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    
    return [t for t in tokens if t not in stop_words and len(t) > 2]


# -----------------------------------------------------------------------------
# 2. HUGGINGFACE SUBWORD TOKENISATION
# -----------------------------------------------------------------------------

def get_huggingface_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Gets a HuggingFace tokenizer securely."""
    if AutoTokenizer is None:
        raise ImportError("pip install transformers to use HuggingFace Subwords.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Patch GPT-2 / Llama style tokenizers missing padding tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def tokenize_texts(tokenizer: PreTrainedTokenizerBase, texts: Union[str, List[str]], max_length: int = 128) -> Dict[str, torch.Tensor]:
    """
    Standard padding & truncation tokenisation returning PyTorch Tensors.
    """
    if isinstance(texts, str):
        texts = [texts]
        
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )


# -----------------------------------------------------------------------------
# 3. SFT MASKING & DYNAMIC COLLATION
# -----------------------------------------------------------------------------

def mask_prompt_tokens(input_ids: torch.Tensor, response_start_idx: int) -> torch.Tensor:
    """
    Set prompt portion to IGNORE_INDEX (-100) so cross-entropy loss skips them 
    during RLHF / Instruction Fine-Tuning.
    """
    labels = input_ids.clone()
    labels[:, :response_start_idx] = IGNORE_INDEX
    return labels


class SFTCollator:
    """
    Dynamic padding collate_fn specifically for Variable-Length Instruction Fine-Tuning.
    Expects items to have dicts containing `input_ids` and optionally `labels`.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"].squeeze(0) for item in batch]
        
        # Determine labels logic
        if "labels" in batch[0]:
            labels_list = [item["labels"].squeeze(0) for item in batch]
        else:
            labels_list = [item["input_ids"].squeeze(0).clone() for item in batch]
        
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels    = pad_sequence(labels_list,    batch_first=True, padding_value=IGNORE_INDEX)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
