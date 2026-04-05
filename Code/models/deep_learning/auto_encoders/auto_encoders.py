import torch
import torch.nn as nn
import os
from typing import Union, Any, Tuple, List, Dict, Callable
from ....models.utils import DLModule
import json
import tqdm
import torch.optim as optim


class AutoEncoderGeneral(DLModule):
    def __init__(self,
                 encoder_type: str,
                 decoder_type: str,
                 encoder_configs: Union[Dict, List[Dict]],
                 decoder_configs: Union[Dict, List[Dict]],
                 latent_size: int = None,
                 seed: int = 42,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.latent_size = latent_size
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.device = device
        self.dtype = dtype

        # Set seed
        torch.manual_seed(seed)
        if hasattr(self, 'set_seed'):
            self.set_seed(seed)

        # Registry Path (assuming standard location)
        self.registry_path = "src/models/__registry__.json"

        # Instantiate Models
        self.encoder = self._build_model(encoder_type, encoder_configs, is_encoder=True)
        self.decoder = self._build_model(decoder_type, decoder_configs, is_encoder=False)

        self.to(device=self.device)

    def _get_model_class(self, model_type: str):
        """
        Dynamically finds the class from registry.
        """
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                raise FileNotFoundError(f"Registry not found at {self.registry_path}")
        except Exception as e:
            print(f"Warning: Registry lookup failed ({e}). Using fallback map.")
            return self._fallback_model_lookup(model_type)

        # Structure: [ {"DeepLearning": { "Category": { "FileKey": [Classes] } } } ]
        deep_learning = registry[0].get("DeepLearning", {})

        target_class = None
        target_file_key = None

        for category, modules in deep_learning.items():
            for file_key, classes in modules.items():
                if model_type in classes:
                    target_class = model_type
                    target_file_key = file_key
                    break
            if target_class:
                break

        if not target_class:
            raise ValueError(f"Model type '{model_type}' not found in registry.")

        module_path = self._resolve_module_path(target_file_key)

        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, target_class)
            return cls
        except Exception as e:
            raise ImportError(f"Failed to import class '{target_class}' from '{module_path}': {e}")

    def _resolve_module_path(self, file_key):
        """Known mappings including transformer modules for future use."""
        mappings = {
            "rnn_family": "Code.models.deep_learning.rnn",
            "llm_models": "Code.models.deep_learning.llm.llm_models",
            "nn_models": "Code.models.deep_learning.ffnn.nn_models",
            "cnn_models": "Code.models.deep_learning.cnn.models",
            "gan_models": "Code.models.deep_learning.gan.gan_models",
            "auto_encoders": "Code.models.deep_learning.auto_encoders.auto_encoders",
            "transformer_models": "Code.models.deep_learning.transformers.transformers.transformers",
            "transformer_decoders": "Code.models.deep_learning.transformers.transformers.decoders.decoders",
            "models": "Code.models.deep_learning.models"
        }

        if file_key in mappings:
            return mappings[file_key]

        raise ValueError(f"Unknown registry file key: {file_key}. Add to _resolve_module_path.")

    def _fallback_model_lookup(self, model_type):
        """Fallback import when registry lookup fails."""
        if "FeedForward" in model_type:
            from ....models.deep_learning.ffnn.nn_models import FeedForwardNeuralNetwork
            return FeedForwardNeuralNetwork
        if "CNN" in model_type or "CNNetworks" in model_type:
            from ....models.deep_learning.cnn.models import CNNetworks
            return CNNetworks
        if "Transformer" in model_type:
            try:
                import importlib
                mod = importlib.import_module(
                    "Code.models.deep_learning.transformers.transformers.transformers"
                )
                if hasattr(mod, model_type):
                    return getattr(mod, model_type)
            except (ImportError, AttributeError):
                pass
            try:
                import importlib
                mod = importlib.import_module(
                    "Code.models.deep_learning.transformers.transformers.decoders.decoders"
                )
                if hasattr(mod, model_type):
                    return getattr(mod, model_type)
            except (ImportError, AttributeError):
                pass
            raise ImportError(
                f"Transformer model '{model_type}' not yet implemented. "
                "Transformers are being built - add the class to transformers/transformers/transformers.py "
                "or transformers/transformers/decoders/decoders.py when ready."
            )
        raise ValueError(f"Could not resolve model '{model_type}' via fallback.")

    def _build_model(self, model_type, config, is_encoder=True):
        cls = self._get_model_class(model_type)

        if isinstance(config, (list, tuple)):
            return cls(*config, device=self.device, dtype=self.dtype)
        elif isinstance(config, dict):
            return cls(**config, device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Config must be dict (kwargs) or list (args).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode (reconstruction).
        """
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def fit(self,
            data: Union[torch.utils.data.DataLoader, Any],
            epochs: int = 1,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            optimizer: Union[Callable, str] = 'adam',
            loss: Union[Callable, str] = 'mse',
            betas: Tuple[float, float] = (0.9, 0.999),
            show_progress_bar: bool = True,
            save_dir: str = 'checkpoints/auto_encoder',
            save_type: str = 'pt',
            **kwargs):
        """
        AutoEncoder training loop with reconstruction loss.
        """
        # 1. Setup Data
        if not isinstance(data, torch.utils.data.DataLoader):
            if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                from torch.utils.data import DataLoader
                dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
            else:
                raise ValueError("Data must be a DataLoader or Dataset-like object.")
        else:
            dataloader = data
            if dataloader.batch_size is not None:
                batch_size = dataloader.batch_size

        # 2. Setup Device
        self.to(self.device)
        print(f"Training AutoEncoder on {self.device}...")

        # 3. Setup Loss
        if isinstance(loss, str):
            if loss.lower() == 'mse':
                criterion = nn.MSELoss()
            elif loss.lower() == 'bce':
                criterion = nn.BCELoss()
            elif loss.lower() == 'l1' or loss.lower() == 'mae':
                criterion = nn.L1Loss()
            else:
                criterion = nn.MSELoss()
        else:
            criterion = loss

        # 4. Setup Optimizer (single optimizer for encoder + decoder)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                opt = optim.Adam(params, lr=learning_rate, betas=betas)
            elif optimizer.lower() == 'sgd':
                opt = optim.SGD(params, lr=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                opt = optim.RMSprop(params, lr=learning_rate)
            else:
                opt = optim.Adam(params, lr=learning_rate, betas=betas)
        elif callable(optimizer):
            opt = optimizer(params, lr=learning_rate)
        else:
            opt = optimizer

        # 5. Training Loop
        history = []

        for epoch in range(epochs):
            if show_progress_bar:
                pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                pbar = dataloader

            epoch_losses = []

            for i, batch in enumerate(pbar):
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    keys = ['image', 'images', 'img', 'x', 'input_ids', 'data']
                    x = None
                    for k in keys:
                        if k in batch:
                            x = batch[k]
                            break
                    if x is None:
                        x = list(batch.values())[0]
                else:
                    x = batch

                x = x.to(self.device)

                opt.zero_grad()
                x_recon = self.forward(x)
                loss_val = criterion(x_recon, x)
                loss_val.backward()
                opt.step()

                epoch_losses.append(loss_val.item())

                if show_progress_bar:
                    pbar.set_postfix({'loss': loss_val.item()})

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            history.append({'epoch': epoch + 1, 'loss': avg_loss})

            # Save Checkpoint
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.encoder.state_dict(),
                           os.path.join(save_dir, f"encoder_epoch_{epoch + 1}.{save_type}"))
                torch.save(self.decoder.state_dict(),
                           os.path.join(save_dir, f"decoder_epoch_{epoch + 1}.{save_type}"))

        return history
