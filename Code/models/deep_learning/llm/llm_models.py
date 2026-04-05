import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
import os
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Tuple, Callable, Optional

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    sklearn_available = True
except ImportError:
    sklearn_available = False
    warnings.warn("`sklearn` not found. Dimensionality reduction and clustering features will be disabled.")

from ....models.deep_learning.models import DLModelLayers
from ....models.utils import DLModule
from ....models.deep_learning.transformers.models.huggingface import HuggingFaceTransformer


def _infer_last_linear_out_features(
    layers: Optional[Union[List[Tuple[str, dict]], Dict[str, dict]]],
) -> Optional[int]:
    """Output width of the last linear-like layer in ``layers`` (for MLP heads)."""
    if layers is None:
        return None
    seq: List[Tuple[str, dict]]
    if isinstance(layers, dict):
        seq = [(k, v) for k, v in layers.items() if isinstance(v, dict)]
    else:
        seq = [(str(lt), cfg) for lt, cfg in layers if isinstance(cfg, dict)]
    for lt, cfg in reversed(seq):
        name = lt.lower()
        if name in (
            "linear",
            "dense",
            "nn",
            "lazy_nn",
            "lazy_linear",
            "lazy_dense",
        ):
            od = cfg.get("out_features")
            if od is not None:
                return int(od)
    return None


class LLMModule(DLModule):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 n_classes: int,
                 layers: Optional[Union[List[Tuple[str, dict]], Dict[str, dict]]] = None,
                 act_funcs: Optional[
                     Union[List[Union[nn.Module, Callable]], Dict[str, Union[nn.Module, Callable]], Tuple[
                         Union[nn.Module, Callable], ...]]] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 checkpoint_dir: str = "checkpoints/deep_learning/llm",
                 single_linear_head: bool = True,
                 embed_dim: int = 768,
                 l2_normalize_pooled: bool = False,
                 *args, **kwargs):
        super().__init__()

        default_model = "google-bert/bert-base-uncased"

        self.device = device
        self.dtype = dtype
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            except OSError:
                fb = Path(tempfile.gettempdir()) / "thesis_llm_hf_cache"
                fb.mkdir(parents=True, exist_ok=True)
                self.checkpoint_dir = str(fb)

        self._hf_wrapper: Optional[HuggingFaceTransformer] = None
        self.model = None
        self.tokenizer = None
        self.model_type = None

        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        local_model_path = os.path.join(self.checkpoint_dir, safe_model_name)

        # ── 1. Try loading from local checkpoint ─────────────────────
        if os.path.exists(local_model_path):
            for _mtype, _internal in (("auto", "transformers"), ("sentence_transformer", "sentence_transformers")):
                try:
                    self._hf_wrapper = HuggingFaceTransformer.from_pretrained(
                        local_model_path,
                        model_type=_mtype,
                        tokenizer_id=local_model_path,
                        device=device,
                        dtype=dtype,
                    )
                    self.model_type = _internal
                    print(f"Loaded model '{model_name}' from local checkpoint: {local_model_path}")
                    break
                except Exception:
                    self._hf_wrapper = None

        # ── 2. Download from HF Hub ───────────────────────────────────
        if self._hf_wrapper is None:
            tok_id = tokenizer_name if tokenizer_name else model_name
            for _mtype, _internal in (("auto", "transformers"), ("sentence_transformer", "sentence_transformers")):
                try:
                    self._hf_wrapper = HuggingFaceTransformer.from_pretrained(
                        model_name,
                        model_type=_mtype,
                        tokenizer_id=tok_id,
                        device=device,
                        dtype=dtype,
                    )
                    self.model_type = _internal
                    print(f"Downloading and saving model '{model_name}' to {local_model_path}...")
                    self._hf_wrapper.save_pretrained(local_model_path)
                    break
                except Exception:
                    self._hf_wrapper = None

        # ── 3. Fallback to default model ─────────────────────────────
        if self._hf_wrapper is None:
            warnings.warn(
                f"Couldn't load model '{model_name}'. Switching to default '{default_model}'."
            )
            safe_default_name = default_model.replace("/", "_")
            local_default_path = os.path.join(self.checkpoint_dir, safe_default_name)
            load_path = local_default_path if os.path.exists(local_default_path) else default_model
            try:
                self._hf_wrapper = HuggingFaceTransformer.from_pretrained(
                    load_path,
                    model_type="auto",
                    device=device,
                    dtype=dtype,
                )
                self.model_type = "transformers"
                if not os.path.exists(local_default_path):
                    print(f"Saving default model to {local_default_path}...")
                    self._hf_wrapper.save_pretrained(local_default_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load default model '{default_model}': {exc}"
                ) from exc

        # ── Expose raw handles used by task layers ────────────────────
        self.model = self._hf_wrapper.hf_model
        self.tokenizer = self._hf_wrapper.hf_tokenizer

        n_classes = int(math.fabs(n_classes))
        self.n_classes = n_classes
        self.single_linear_head = single_linear_head
        self.embed_dim = int(embed_dim)
        self.l2_normalize_pooled = l2_normalize_pooled

        # Training uses raw logits + CrossEntropyLoss; probabilities at inference via predict_proba.
        self.final_act = torch.sigmoid if self.n_classes == 2 else lambda z: torch.softmax(z, dim=-1)

        if self.single_linear_head:
            self.classifier = nn.Linear(self.embed_dim, self.n_classes, bias=True, device=device, dtype=dtype)
            self.module = None
            self.final_layer = None
        else:
            if layers is None:
                raise ValueError("layers required when single_linear_head=False")
            self.classifier = None
            self.module = DLModelLayers(layers, act_funcs, device=device, dtype=dtype, *args, **kwargs)
            # LazyLinear breaks DistributedDataParallel until a forward pass; use nn.Linear when width is known.
            _tail = _infer_last_linear_out_features(layers)
            if _tail is not None:
                self.final_layer = nn.Linear(
                    _tail, self.n_classes, bias=False, device=device, dtype=dtype
                )
            else:
                self.final_layer = nn.LazyLinear(
                    self.n_classes, bias=False, device=device, dtype=dtype
                )

        if self.model:
            self.model.to(device)
            if self.model_type == 'transformers':
                self.model.to(dtype)
            # Default: Freeze backbone
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool = True):
        """Sets the backbone model's parameters to be trainable or frozen."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = trainable

    def _backbone_device(self) -> torch.device:
        """Device of HF backbone weights (matches after .to(cuda) / DDP); not the init JSON ``device`` string."""
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device(self.device if self.device else "cpu")

    def tokenize_text(self, text: Union[str, List[str], Tuple[str]]) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        # Tokenizer returns integers (input_ids), do NOT cast to self.dtype (float)
        return self.tokenizer(text, padding=True, truncation=True,
                              return_tensors='pt').to(device=self._backbone_device())

    def get_embeddings(self, 
                       texts: Union[str, List[str], Tuple[str]], 
                       pooling_strategy: str = 'mean', 
                       layer_strategy: str = 'last',
                       apply_l2_normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Generates embeddings based on pooling and layer strategies.
        pooling_strategy: 'cls', 'mean', 'max'
        layer_strategy: 'last', 'concat_last_4', 'mean_last_4' (transformers only)
        """
        if self.model_type == 'transformers':
            encoded_inputs = self.tokenize_text(texts)
            output_hidden_states = (layer_strategy in ['concat_last_4', 'mean_last_4'])
            
            # Use torch.no_grad only if not training or if backbone is explicitly frozen
            is_trainable = any(p.requires_grad for p in self.model.parameters())
            context_manager = torch.no_grad() if not (self.training and is_trainable) else torch.enable_grad()
            
            with context_manager:
                outputs = self.model(**encoded_inputs, output_hidden_states=output_hidden_states)
            
            # Extract Hidden States
            if layer_strategy == 'concat_last_4':
                # Stack last 4 layers: [Batch, Seq, Hidden*4]
                # outputs.hidden_states is tuple of (Batch, Seq, Hidden)
                # Last one is outputs.last_hidden_state
                # We need last 4
                hidden_states = outputs.hidden_states[-4:]
                # Concatenate along hidden dimension
                token_embeddings = torch.cat(hidden_states, dim=-1) # [B, L, 4*H]
            elif layer_strategy == 'mean_last_4':
                hidden_states = outputs.hidden_states[-4:]
                token_embeddings = torch.stack(hidden_states).mean(dim=0) # [B, L, H]
            else:
                token_embeddings = outputs.last_hidden_state # [B, L, H]

            attn_mask = encoded_inputs['attention_mask'] # [B, L]
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            if pooling_strategy == 'cls':
                # CLS token is usually at index 0
                embeddings = token_embeddings[:, 0, :]
            elif pooling_strategy == 'max':
                # Mask padding tokens with very small value before max
                token_embeddings[input_mask_expanded == 0] = -1e9
                embeddings = torch.max(token_embeddings, 1)[0]
            else: # mean (default)
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            do_norm = self.l2_normalize_pooled if apply_l2_normalize is None else apply_l2_normalize
            if do_norm:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.to(dtype=self.dtype)
            return embeddings

        elif self.model_type == 'sentence_transformers':
            # SentenceTransformers handles pooling internally usually
            out = self.model.encode(texts, convert_to_tensor=True, device=self._backbone_device())
            out = out.to(dtype=self.dtype)
            return out
        else:
             raise RuntimeError("Model model not initialized.")

    def reduce_dimensions(self, embeddings: torch.Tensor, method: str = 'pca', n_components: int = 2) -> torch.Tensor:
        # Primary: use machine_learning.transformer package
        try:
            from ....models.machine_learning.transformer import PCA as MLPCA, TSNE as MLTSNE
            if method == 'pca':
                reducer = MLPCA(n_components=n_components, device=self.device, dtype=self.dtype)
            elif method == 'tsne':
                reducer = MLTSNE(n_components=n_components, device=self.device, dtype=self.dtype)
            else:
                reducer = MLPCA(n_components=n_components, device=self.device, dtype=self.dtype)
            X = embeddings.detach().to(device=self.device, dtype=self.dtype)
            reduced = reducer.fit_transform(X)
            return reduced.to(device=self.device, dtype=self.dtype)
        except (ImportError, Exception):
            pass  # fallback to sklearn

        # Fallback: sklearn
        if not sklearn_available:
            warnings.warn("Neither transformer nor sklearn available. Returning original embeddings.")
            return embeddings

        embeddings_np = embeddings.detach().cpu().numpy()
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        else:
            reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings_np)
        return torch.tensor(reduced, device=self.device, dtype=self.dtype)

    def cluster_embeddings(self, embeddings: torch.Tensor, method: str = 'kmeans', n_clusters: int = 2) -> torch.Tensor:
        # Primary: use machine_learning.clustering package
        try:
            from ....models.machine_learning.clustering import (
                KMeansCluster,
                DBSCAN as MLDBSCAN,
                AgglomerativeClustering as MLAgglomerativeClustering,
            )
            X = embeddings.detach().to(device=self.device, dtype=self.dtype)
            if method == 'kmeans':
                clusterer = KMeansCluster(n_clusters=n_clusters, n_init='auto', device=self.device, dtype=self.dtype)
            elif method == 'dbscan':
                clusterer = MLDBSCAN(device=self.device, dtype=self.dtype)
            elif method == 'agglomerative':
                clusterer = MLAgglomerativeClustering(n_clusters=n_clusters, device=self.device, dtype=self.dtype)
            else:
                clusterer = KMeansCluster(n_clusters=n_clusters, n_init='auto', device=self.device, dtype=self.dtype)
            labels = clusterer.fit_predict(X)
            return labels.to(device=self.device, dtype=torch.long)
        except (ImportError, Exception):
            pass  # fallback to sklearn

        # Fallback: sklearn
        if not sklearn_available:
            warnings.warn("Neither clustering nor sklearn available. Returning zeros.")
            return torch.zeros(embeddings.size(0), device=self.device, dtype=torch.long)

        embeddings_np = embeddings.detach().cpu().numpy()
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
        elif method == 'dbscan':
            clusterer = DBSCAN()
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clusterer = KMeans(n_clusters=n_clusters, n_init='auto')
        labels = clusterer.fit_predict(embeddings_np)
        return torch.tensor(labels, device=self.device, dtype=torch.long)

    def forward(self, 
                texts: Union[str, List[str], Tuple[str]],
                return_type: str = 'logits',  # 'logits', 'embeddings', 'clusters', 'reduced'
                pooling_strategy: str = 'mean',
                layer_strategy: str = 'last',
                dim_reduction_method: str = 'pca',
                n_components: int = 2,
                cluster_method: str = 'kmeans',
                n_clusters: int = 2) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if return_type == 'logits':
             embeddings = self.get_embeddings(
                 texts,
                 pooling_strategy=pooling_strategy,
                 layer_strategy=layer_strategy,
             )
             if self.single_linear_head:
                 return self.classifier(embeddings)
             out = self.module(embeddings)
             out = self.final_layer(out)
             return out
        
        # New features
        embeddings = self.get_embeddings(texts, pooling_strategy=pooling_strategy, layer_strategy=layer_strategy)
        
        if return_type == 'embeddings':
            return embeddings
            
        if return_type == 'reduced':
            return self.reduce_dimensions(embeddings, method=dim_reduction_method, n_components=n_components)
            
        if return_type == 'clusters':
            return self.cluster_embeddings(embeddings, method=cluster_method, n_clusters=n_clusters)
            
        if return_type == 'all':
            res = {'embeddings': embeddings}
            res['reduced'] = self.reduce_dimensions(embeddings, method=dim_reduction_method, n_components=n_components)
            res['clusters'] = self.cluster_embeddings(embeddings, method=cluster_method, n_clusters=n_clusters)
            return res

        return embeddings # Default fallback
