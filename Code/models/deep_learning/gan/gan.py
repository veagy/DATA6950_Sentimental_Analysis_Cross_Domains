import torch
import torch.nn as nn
import os
from typing import Union, Any, Tuple, List, Dict, Callable
from ....models.utils import DLModule
import json
import tqdm
import torch.optim as optim


class GANGeneral(DLModule):
    def __init__(self,
                 latent_size: int,
                 generator_type: str,
                 discriminator_type: str,
                 generator_configs: Union[Dict, List[Dict]],
                 discriminator_configs: Union[Dict, List[Dict]],
                 seed: int = 42,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        super().__init__()
        self.latent_size = latent_size
        self.generator_type = generator_type
        self.discriminator_type = discriminator_type
        self.device = device
        self.dtype = dtype

        # Set seed
        torch.manual_seed(seed)
        if hasattr(self, 'set_seed'):  # If DLModule has this
            self.set_seed(seed)

        # Registry Path (assuming standard location)
        self.registry_path = "src/models/__registry__.json"

        # Instantiate Models
        self.generator = self._build_model(generator_type, generator_configs, is_generator=True)
        self.discriminator = self._build_model(discriminator_type, discriminator_configs, is_generator=False)

        self.to(device=self.device)

    def _get_model_class(self, model_type: str):
        """
        Dynamically finds the class from registry.
        """
        # 1. Load Registry
        try:
            # We need absolute path or relative from cwd
            # Assuming CWD is project root
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                # Fallback: Hardcoded paths for now if file not found
                # Or try to construct path relative to file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # ../../../../src/models/__registry__.json ?
                # Let's hope CWD is root or registry is in standard place
                raise FileNotFoundError(f"Registry not found at {self.registry_path}")
        except Exception as e:
            # Fallback map for known types to avoid registry dependency if it fails
            print(f"Warning: Registry lookup failed ({e}). Using fallback map.")
            return self._fallback_model_lookup(model_type)

        # 2. Search Registry
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
            if target_class: break

        if not target_class:
            raise ValueError(f"Model type '{model_type}' not found in registry.")

        # 3. Import
        # Map file_key to module path
        # This mapping needs to be maintained or inferred. 
        # Inference: "rnn_family" -> "Code.models.deep_learning.rnn.RNNFamily"? 
        # "nn_models" -> "Code.models.deep_learning.ffnn.nn_models"

        module_path = self._resolve_module_path(target_file_key)

        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, target_class)
            return cls
        except Exception as e:
            raise ImportError(f"Failed to import class '{target_class}' from '{module_path}': {e}")

    def _resolve_module_path(self, file_key):
        # Known mappings (including transformer modules for future use)
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
        # basic import attempts
        if "FeedForward" in model_type:
            from ....models.deep_learning.ffnn.nn_models import FeedForwardNeuralNetwork
            return FeedForwardNeuralNetwork
        if "CNN" in model_type:
            from ....models.deep_learning.cnn.models import CNNetworks
            return CNNetworks
        if "Transformer" in model_type or "Encoder" in model_type or "Decoder" in model_type:
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

    def _build_model(self, model_type, config, is_generator=True):
        cls = self._get_model_class(model_type)

        # Prepare kwargs
        # If config is dict, pass as kwargs? 
        # Or does the model init expect specific args?
        # Most of our models take arg lists/dicts (dims, capabilities etc.)

        # If config is a list of args? 
        if isinstance(config, (list, tuple)):
            # Assuming config is list of args
            return cls(*config, device=self.device, dtype=self.dtype)
        elif isinstance(config, dict):
            # Kwargs
            return cls(**config, device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Config must be dict (kwargs) or list (args).")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass usually means "Generate".
        """
        return self.generator(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def fit(self,
            data: Union[torch.utils.data.DataLoader, Any],
            epochs: int = 1,
            batch_size: int = 32,
            learning_rate: float = 0.0002,
            g_optimizer: Union[Callable, str] = 'adam',
            d_optimizer: Union[Callable, str] = 'adam',
            loss: Union[Callable, str] = 'bce',
            betas: Tuple[float, float] = (0.5, 0.999),
            show_progress_bar: bool = True,
            save_dir: str = 'checkpoints/gan',
            save_type: str = 'pt',
            **kwargs):
        """
        GAN-specific training loop.
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
        print(f"Training GAN on {self.device}...")

        # 3. Setup Loss
        if isinstance(loss, str):
            if loss.lower() == 'bce':
                criterion = nn.BCELoss()
            elif loss.lower() == 'mse':
                criterion = nn.MSELoss()
            else:
                criterion = nn.BCELoss()  # Default
        else:
            criterion = loss

        # 4. Setup Optimizers
        # Helper to get optimizer
        def get_opt(opt_type, params, lr, betas):
            if isinstance(opt_type, str):
                if opt_type.lower() == 'adam':
                    return optim.Adam(params, lr=lr, betas=betas)
                elif opt_type.lower() == 'sgd':
                    return optim.SGD(params, lr=lr)
                elif opt_type.lower() == 'rmsprop':
                    return optim.RMSprop(params, lr=lr)
                else:
                    return optim.Adam(params, lr=lr, betas=betas)
            elif isinstance(opt_type, Callable):
                return opt_type(params, lr=lr)
            return opt_type

        opt_g = get_opt(g_optimizer, self.generator.parameters(), learning_rate, betas)
        opt_d = get_opt(d_optimizer, self.discriminator.parameters(), learning_rate, betas)

        # 5. Training Loop
        history = []

        for epoch in range(epochs):
            if show_progress_bar:
                pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                pbar = dataloader

            g_losses = []
            d_losses = []

            for i, batch in enumerate(pbar):
                # Unpack batch
                if isinstance(batch, (tuple, list)):
                    real_images = batch[0]
                elif isinstance(batch, dict):
                    # Try common keys
                    keys = ['image', 'images', 'img', 'x', 'input_ids']
                    real_images = None
                    for k in keys:
                        if k in batch:
                            real_images = batch[k]
                            break
                    if real_images is None:
                        # Fallback: take first value?
                        real_images = list(batch.values())[0]
                else:
                    real_images = batch

                real_images = real_images.to(self.device)

                # If images are not flat and models handle them, great. 
                # Otherwise user ensures compatibility.
                current_batch_size = real_images.size(0)

                # Labels
                real_label = torch.ones(current_batch_size, 1, device=self.device)
                fake_label = torch.zeros(current_batch_size, 1, device=self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                opt_d.zero_grad()

                # Train with Real
                output_real = self.discriminator(real_images)

                # Handle output shape: if D outputs [B, 1] usually. 
                # Sometimes models output [B, C] logits. 
                # If Criterion is BCE, expects probabilities if BCELoss, logits if BCEWithLogits.
                # Assuming BCELoss and Sigmoid activation in D usually, or handled in model.
                # If generic model (e.g. FFNN) used as D, it might not have Sigmoid at end unless configured.
                # We can add Sigmoid if needed or check range? 
                # For robust GAN, usually we let the model handle activation.

                # Check shape compatibility
                if output_real.shape != real_label.shape:
                    # Try to squeeze/reshape? 
                    # If output is [B, 2] (classes), then real_label should be class index? 
                    # Standard GAN D outputs scalar probability/logit. 
                    # Let's assume [B, 1] or [B].
                    output_real = output_real.view(current_batch_size, -1)
                    if output_real.shape[1] > 1:
                        # Softmax/Class based? 
                        # This fit method assumes standard Binary GAN.
                        pass

                errD_real = criterion(output_real, real_label)
                errD_real.backward()

                # Train with Fake
                noise = torch.randn(current_batch_size, self.latent_size, device=self.device)
                fake_images = self.generator(noise)

                # Detach fake images so we don't backprop to G yet
                output_fake = self.discriminator(fake_images.detach())
                output_fake = output_fake.view(current_batch_size, -1)

                errD_fake = criterion(output_fake, fake_label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                opt_d.step()

                # -----------------
                #  Train Generator
                # -----------------
                opt_g.zero_grad()

                # G wants D to think fake images are real
                output_fake_for_g = self.discriminator(fake_images)  # Re-compute or use saved graph?
                # Note: `fake_images` was computed above. If we didn't detach `fake_images` variable itself but passed `fake_images.detach()` to D, 
                # then `fake_images` still has graph for G. 
                # Yes, we pass `fake_images` (with graph) to D here.

                output_fake_for_g = output_fake_for_g.view(current_batch_size, -1)

                errG = criterion(output_fake_for_g, real_label)  # We want D to predict Real via G's parameters
                errG.backward()
                opt_g.step()

                # Logging
                g_losses.append(errG.item())
                d_losses.append(errD.item())

                if show_progress_bar:
                    pbar.set_postfix({'g_loss': errG.item(), 'd_loss': errD.item()})

            avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else 0.0
            avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0.0
            history.append({'epoch': epoch + 1, 'g_loss': avg_g_loss, 'd_loss': avg_d_loss})

            # Save Checkpoint
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Save both models
                torch.save(self.generator.state_dict(),
                           os.path.join(save_dir, f"generator_epoch_{epoch + 1}.{save_type}"))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(save_dir, f"discriminator_epoch_{epoch + 1}.{save_type}"))

        return history
