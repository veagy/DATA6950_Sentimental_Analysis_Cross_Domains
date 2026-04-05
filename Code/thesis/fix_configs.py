import os
import json

ROOT = r"d:\CAPSTONE\capstone-2"
config_dir = os.path.join(ROOT, "Code", "thesis", "config")

def fix_json_configs():
    for root, _, files in os.walk(config_dir):
        for file in files:
            if not file.endswith(".json"): continue
            path = os.path.join(root, file)
            with open(path, "r") as f:
                config = json.load(f)
            
            class_name = list(config.keys())[0]
            
            if class_name in ["CNNetworks", "CNNetworksOp"]:
                config_new = {
                    class_name: {
                        "dimensionality": 1,
                        "layer_types": ["conv", "pool", "fc", "act", "fc"],
                        "channels": [100, 400, 150, 3 if "3_labels" in str(path) else 2],
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
                        "device": "cpu"
                    }
                }
                config = config_new
            
            elif class_name == "LSTMModule" and "B10" in file:
                config_new = {
                    "LSTMModule": {
                        "input_size": 100,
                        "hidden_size": 275,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                        "proj_size": 0,
                        "device": "cpu"
                    }
                }
                config = config_new
                
            elif class_name == "GRUModule":
                config_new = {
                    "GRUModule": {
                        "input_size": 100,
                        "hidden_size": 380,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                        "device": "cpu"
                    }
                }
                config = config_new
                
            elif class_name == "LSTMModule" and "B7" in file:
                config_new = {
                    "LSTMModule": {
                        "input_size": 100,
                        "hidden_size": 260,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": True,
                        "proj_size": 0,
                        "device": "cpu"
                    }
                }
                config = config_new
                
            elif class_name == "LogisticRegression":
                config_new = {
                    "LogisticRegression": {
                        "C": 1.0,
                        "fit_intercept": True,
                        "solver": "lbfgs",
                        "max_iter": 100
                    }
                }
                config = config_new
                
            elif class_name == "HierarchicalReasoningModule" or class_name == "HierarchicalReasoningModel":
                config_new = {
                    "HierarchicalReasoningModel": {
                        "config": {
                            "batch_size": 8,
                            "seq_len": 128,
                            "hidden_size": 768,
                            "H_cycles": 1,
                            "L_cycles": 1,
                            "h_level_model": "EncoderLM",
                            "l_level_model": "EncoderLM",
                            "tokenizer_name": "google-bert/bert-base-uncased",
                            "model_kwargs": {"vocab_size": 30522, "num_layers": 1, "num_heads": 4}
                        },
                        "n_classes": 3 if "3_labels" in str(path) else 2
                    }
                }
                config = config_new
                
            elif class_name == "LLMModule":
                n_classes = 3 if "3_labels" in str(path) else 2
                model_name = "distilbert-base-uncased"
                if "B4" in file: model_name = "google-bert/bert-base-uncased"
                if "B5" in file: model_name = "roberta-base"
                if "B6" in file: model_name = "facebook/bart-base"
                
                config_new = {
                    "LLMModule": {
                        "model_name": model_name,
                        "tokenizer_name": model_name,
                        "n_classes": n_classes,
                        "layers": ["linear", {"in_features": 768, "out_features": 256}],
                        "act_funcs": ["ReLU"],
                        "device": "cpu"
                    }
                }
                config = config_new

            elif "B11_CNN_LSTM_v1" in file:
                config_new = {
                    class_name: {
                        "dimensionality": 1,
                        "layer_types": ["conv", "pool", "fc", "act", "fc"],
                        "channels": [100, 800, 300, 3 if "3_labels" in str(path) else 2],
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
                        "device": "cpu"
                    }
                }
                config = config_new

            with open(path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Fixed {path}")

if __name__ == "__main__":
    fix_json_configs()
