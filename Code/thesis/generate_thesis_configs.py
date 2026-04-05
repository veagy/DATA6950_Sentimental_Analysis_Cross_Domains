"""One-off generator for thesis JSON configs (run from repo root if needed)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (capstone-2)
CFG = Path(__file__).resolve().parent / "config"


def write(rel: str, obj: dict) -> None:
    p = CFG / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def llm(name: str, tok: str, n_class: int, embed: int = 768) -> dict:
    return {
        "LLMModule": {
            "model_name": name,
            "tokenizer_name": tok,
            "n_classes": n_class,
            "single_linear_head": True,
            "embed_dim": embed,
            "l2_normalize_pooled": False,
            "checkpoint_dir": str(ROOT / "checkpoints" / "deep_learning" / "llm"),
            "device": "cpu",
        }
    }


def hrm_encoder(n_layers: int = 4) -> dict:
    """Shared E-HRM1 encoder JSON; use --n_classes 2 or 3 when invoking train_single.py."""
    return {
        "HierarchicalReasoningModel": {
            "config": {
                "batch_size": 8,
                "seq_len": 512,
                "hidden_size": 800,
                "output_embed_dim": 100,
                "vocab_size": 30522,
                "H_cycles": 2,
                "L_cycles": 2,
                "h_level_model": "EncoderLM",
                "l_level_model": "EncoderLM",
                "tokenizer_name": "google-bert/bert-base-uncased",
                "model_kwargs": {"num_layers": n_layers, "num_heads": 16},
            },
        }
    }


def cnn(n_class: int) -> dict:
    return {
        "CNNetworks": {
            "dimensionality": 1,
            "layer_types": ["conv", "pool", "fc", "act", "fc"],
            "channels": [100, 400, 150, n_class],
            "kernel_size": [3, 2],
            "stride": [1, 2],
            "padding": [1, 0],
            "pool_type": ["max"],
            "act_funcs": ["ReLU"],
            "pad_type": [],
            "dropout_percent": [],
            "dilation": [1, 1],
            "groups": [1, 1],
            "bias": [True, True, True],
            "padding_mode": ["zeros", "zeros"],
            "lazy": False,
            "transpose": False,
            "device": "cpu",
        }
    }


def cnn_lstm(n_class: int) -> dict:
    # CNNetworks (not Op): same conv stack as B9; Op linear in_features mismatch with 100D mock input.
    return {
        "CNNetworks": {
            "dimensionality": 1,
            "layer_types": ["conv", "pool", "fc", "act", "fc"],
            "channels": [100, 400, 150, n_class],
            "kernel_size": [3, 2],
            "stride": [1, 2],
            "padding": [1, 0],
            "pool_type": ["max"],
            "act_funcs": ["ReLU"],
            "pad_type": [],
            "dropout_percent": [],
            "dilation": [1, 1],
            "groups": [1, 1],
            "bias": [True, True, True],
            "padding_mode": ["zeros", "zeros"],
            "lazy": False,
            "transpose": False,
            "device": "cpu",
        }
    }


def lstm(hidden: int, bidir: bool, n_in: int = 100) -> dict:
    return {
        "LSTMModule": {
            "input_size": n_in,
            "hidden_size": hidden,
            "num_layers": 1,
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": bidir,
            "proj_size": 0,
            "device": "cpu",
        }
    }


def gru(hidden: int, bidir: bool) -> dict:
    return {
        "GRUModule": {
            "input_size": 100,
            "hidden_size": hidden,
            "num_layers": 1,
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": bidir,
            "device": "cpu",
        }
    }


def main() -> None:
    for n, sub in ((2, "2_labels"), (3, "3_labels")):
        write(f"transformers/{sub}/B3_E_DL1_DistilBERT.json", llm("distilbert-base-uncased", "distilbert-base-uncased", n))
        write(f"transformers/{sub}/B4_E_DL3_BERT.json", llm("google-bert/bert-base-uncased", "google-bert/bert-base-uncased", n))
        write(f"transformers/{sub}/B5_E_DL2_RoBERTa.json", llm("roberta-base", "roberta-base", n))
        write(f"transformers/{sub}/B6_BART.json", llm("facebook/bart-base", "facebook/bart-base", n))
        write(f"cnn/{sub}/B9_CNN_Text.json", cnn(n))
        write(f"rnn/{sub}/B10_LSTM.json", lstm(420, False))
        write(f"rnn/{sub}/B7_BiLSTM.json", lstm(375, True))
        write(f"rnn/{sub}/B8_GRU.json", gru(275, False))
        write(f"rnn/{sub}/B11_CNN_LSTM_v1.json", cnn_lstm(n))
        write(f"ml/{sub}/E_ML1_LogisticRegression.json", {"LogisticRegression": {"C": 1.0, "fit_intercept": True, "solver": "lbfgs", "max_iter": 500}})
        write(
            f"ml/{sub}/E_ML2_LinearSVC.json",
            {"LinearSVC": {"C": 1.0, "max_iter": 2000, "device": "cpu"}},
        )

    write("hrm/E_HRM1_4Level.json", hrm_encoder(n_layers=4))


if __name__ == "__main__":
    main()
    print("Wrote configs under", CFG)
