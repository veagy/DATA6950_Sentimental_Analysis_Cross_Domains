import warnings
import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal, Iterable
from .....models.utils import MLModule
import numpy as np
from torch.func import vmap
import joblib


__all__ = ["PatchExtractor"]


class PatchExtractor(MLModule):
    def __init__(self,
                 patch_size: torch.Tensor = None,
                 max_patches: Union[int, float] = None,
                 random_state: Union[int, torch.Generator] = None,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.random_state = random_state
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        # PatchExtractor is essentially stateless (fit is a no-op)
        self.fit_status = True
        # Attributes set after first transform
        self.patch_size_: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_patch_size(self, img_h: int, img_w: int) -> Tuple[int, int]:
        """Compute concrete patch size from self.patch_size or default rule."""
        if self.patch_size is None:
            ph = max(1, img_h // 10)
            pw = max(1, img_w // 10)
            return (ph, pw)

        ps = self.patch_size
        if isinstance(ps, (list, tuple)) and len(ps) >= 2:
            return (int(ps[0]), int(ps[1]))
        if isinstance(ps, torch.Tensor):
            flat = ps.flatten()
            if flat.numel() >= 2:
                return (int(flat[0].item()), int(flat[1].item()))
            if flat.numel() == 1:
                s = int(flat[0].item())
                return (s, s)
        if isinstance(ps, (int, float)):
            s = int(ps)
            return (s, s)
        raise ValueError(
            "patch_size must be None, an int, or a (height, width) tuple/tensor."
        )

    def _make_generator(self) -> Optional[torch.Generator]:
        """Build a torch.Generator from random_state."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.random_state))
        return g

    def _extract_patches_one(
        self,
        img: torch.Tensor,
        patch_h: int,
        patch_w: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if img.dim() == 2:
            has_channels = False
            img_h, img_w = img.shape
        elif img.dim() == 3:
            has_channels = True
            c, img_h, img_w = img.shape
        else:
            raise ValueError(
                f"Each image must be 2-D (H, W) or 3-D (C, H, W); got shape {tuple(img.shape)}"
            )

        if patch_h > img_h or patch_w > img_w:
            raise ValueError(
                f"Patch size ({patch_h}, {patch_w}) is larger than image size ({img_h}, {img_w})."
            )

        n_rows = img_h - patch_h + 1
        n_cols = img_w - patch_w + 1
        total_patches = n_rows * n_cols

        # Build all patch top-left coordinates
        row_starts = torch.arange(n_rows, device=self.device)
        col_starts = torch.arange(n_cols, device=self.device)
        grid_r, grid_c = torch.meshgrid(row_starts, col_starts, indexing="ij")
        positions = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)  # (T, 2)

        # Determine how many patches to keep
        n_keep = total_patches
        if self.max_patches is not None:
            if isinstance(self.max_patches, float):
                if 0.0 < self.max_patches < 1.0:
                    n_keep = max(1, int(total_patches * self.max_patches))
                else:
                    raise ValueError(
                        "max_patches as float must be in (0, 1)."
                    )
            else:
                n_keep = max(1, min(int(self.max_patches), total_patches))

        if n_keep < total_patches:
            # Random sampling without replacement
            perm = torch.randperm(total_patches, device=self.device, generator=generator)
            positions = positions[perm[:n_keep]]

        # Collect patches
        patches: List[torch.Tensor] = []
        for r, c in positions:
            r_i, c_i = r.item(), c.item()
            if has_channels:
                patch = img[:, r_i: r_i + patch_h, c_i: c_i + patch_w]
            else:
                patch = img[r_i: r_i + patch_h, c_i: c_i + patch_w]
            patches.append(patch)

        return torch.stack(patches, dim=0)  # (n_keep, [C,] H_p, W_p)

    def _load_images(self, X: Any) -> List[torch.Tensor]:
        """Convert various input types to a list of image tensors."""
        if isinstance(X, torch.Tensor):
            if X.dim() == 4:
                return [X[i].to(device=self.device, dtype=self.dtype) for i in range(X.shape[0])]
            if X.dim() == 3:
                # Could be (N, H, W) or (C, H, W); treat as (N, H, W)
                return [X[i].to(device=self.device, dtype=self.dtype) for i in range(X.shape[0])]
            if X.dim() == 2:
                return [X.to(device=self.device, dtype=self.dtype)]
        if isinstance(X, np.ndarray):
            t = torch.from_numpy(X).to(device=self.device, dtype=self.dtype)
            return self._load_images(t)
        if isinstance(X, (list, tuple)):
            result = []
            for item in X:
                if isinstance(item, torch.Tensor):
                    result.append(item.to(device=self.device, dtype=self.dtype))
                elif isinstance(item, np.ndarray):
                    result.append(
                        torch.from_numpy(item).to(device=self.device, dtype=self.dtype)
                    )
                else:
                    raise TypeError(f"Unsupported image type: {type(item)}")
            return result
        raise TypeError(f"X must be a Tensor, ndarray, or list thereof; got {type(X)}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> "PatchExtractor":
        self.fit_status = True
        return self

    def transform(
        self,
        X: Any,
        **kwargs,
    ) -> torch.Tensor:
        
        images = self._load_images(X)
        if not images:
            raise ValueError("X contains no images.")

        # Infer patch size from the first image
        first = images[0]
        if first.dim() == 2:
            img_h, img_w = first.shape
        elif first.dim() == 3:
            _, img_h, img_w = first.shape
        else:
            raise ValueError(f"Images must be 2-D or 3-D tensors; got {first.dim()}-D.")

        patch_h, patch_w = self._resolve_patch_size(img_h, img_w)
        self.patch_size_ = (patch_h, patch_w)
        generator = self._make_generator()

        all_patches: List[torch.Tensor] = []
        for img in images:
            patches = self._extract_patches_one(img, patch_h, patch_w, generator)
            all_patches.append(patches)

        return torch.cat(all_patches, dim=0)

    def fit_transform(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Fit (no-op) and transform X."""
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def reconstruct_from_patches(
        self,
        patches: torch.Tensor,
        image_size: Tuple[int, ...],
    ) -> torch.Tensor:
        if self.patch_size_ is None:
            raise RuntimeError(
                "patch_size_ is not set. Call transform() before reconstruct_from_patches()."
            )
        patch_h, patch_w = self.patch_size_

        if len(image_size) == 2:
            img_h, img_w = image_size
            has_channels = False
            c = 1
        elif len(image_size) == 3:
            c, img_h, img_w = image_size
            has_channels = True
        else:
            raise ValueError("image_size must be (H, W) or (C, H, W).")

        n_rows = img_h - patch_h + 1
        n_cols = img_w - patch_w + 1
        n_patches_per_image = n_rows * n_cols

        if patches.dim() == 3:
            n_total, ph, pw = patches.shape
        elif patches.dim() == 4:
            n_total, ch, ph, pw = patches.shape
        else:
            raise ValueError("patches must be 3-D or 4-D.")

        n_images = n_total // n_patches_per_image
        if n_images == 0:
            raise ValueError(
                f"Cannot reconstruct: patches ({n_total}) < n_patches_per_image ({n_patches_per_image})."
            )

        if has_channels:
            recon = torch.zeros(n_images, c, img_h, img_w, device=self.device, dtype=self.dtype)
            counts = torch.zeros(n_images, 1, img_h, img_w, device=self.device, dtype=self.dtype)
        else:
            recon = torch.zeros(n_images, img_h, img_w, device=self.device, dtype=self.dtype)
            counts = torch.zeros(n_images, img_h, img_w, device=self.device, dtype=self.dtype)

        row_starts = list(range(n_rows))
        col_starts = list(range(n_cols))
        positions = [(r, c_) for r in row_starts for c_ in col_starts]

        for img_idx in range(n_images):
            for k, (r, c_) in enumerate(positions):
                patch_idx = img_idx * n_patches_per_image + k
                if patch_idx >= n_total:
                    break
                patch = patches[patch_idx].to(device=self.device, dtype=self.dtype)
                if has_channels:
                    recon[img_idx, :, r: r + patch_h, c_: c_ + patch_w] += patch
                    counts[img_idx, :, r: r + patch_h, c_: c_ + patch_w] += 1
                else:
                    recon[img_idx, r: r + patch_h, c_: c_ + patch_w] += patch
                    counts[img_idx, r: r + patch_h, c_: c_ + patch_w] += 1

        counts = counts.clamp(min=1)
        return recon / counts

    def forward(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.transform(X, **kwargs)
