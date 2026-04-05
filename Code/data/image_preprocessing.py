"""
Phase 4: Modality-Specific Image Preprocessing.
Supports advanced torchvision transforms, Mixup, Cutmix, and ViT Patching.
"""

import math
import torch
import torch.nn.functional as F
try:
    import torchvision.transforms.v2 as T
except ImportError:
    T = None


# -----------------------------------------------------------------------------
# 1. STANDARD CLASSIFICATION PIPELINES
# -----------------------------------------------------------------------------

def get_train_transform(image_size: int = 224, auto_augment: bool = True):
    """
    Standard ImageNet training transform with aggressive augmentations.
    """
    if T is None:
        raise ImportError("torchvision > 0.15 required for transforms.v2")
    
    transforms_list = [
        T.RandomResizedCrop(size=image_size, scale=(0.08, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5)
    ]
    
    if auto_augment:
        transforms_list.append(T.TrivialAugmentWide())
    else:
        transforms_list.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        
    transforms_list.extend([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return T.Compose(transforms_list)


def get_val_transform(image_size: int = 224):
    """
    Deterministic ImageNet validation transform.
    """
    if T is None:
        raise ImportError("torchvision > 0.15 required for transforms.v2")
        
    resize_dim = int(image_size / 0.875)
    return T.Compose([
        T.Resize(resize_dim, antialias=True),
        T.CenterCrop(image_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def compute_mean_std(loader: torch.utils.data.DataLoader):
    """Compute per-channel mean and std over the full training set iteratively."""
    n_pixels  = 0
    ch_sum    = torch.zeros(3)
    ch_sum_sq = torch.zeros(3)
    for imgs, _ in loader:
        B, C, H, W  = imgs.shape
        # Flatten and aggregate sums
        n_pixels   += B * H * W
        ch_sum     += imgs.sum(dim=[0, 2, 3])
        ch_sum_sq  += (imgs ** 2).sum(dim=[0, 2, 3])
    
    mean = ch_sum / n_pixels
    std  = torch.sqrt((ch_sum_sq / n_pixels) - (mean ** 2))
    return mean.tolist(), std.tolist()


# -----------------------------------------------------------------------------
# 2. ADVANCED DATA AUGMENTATIONS
# -----------------------------------------------------------------------------

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Mathematical MixUp interpolation combining two tensors and labels algebraically.
    """
    B = x.size(0)
    lam  = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    perm = torch.randperm(B, device=x.device)
    
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Mathematical CutMix interpolation substituting structural rectangular patches natively.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    B, C, H, W = x.shape
    
    cut_ratio  = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    
    cx = torch.randint(0, W, (1,), device=x.device).item()
    cy = torch.randint(0, H, (1,), device=x.device).item()
    
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    
    perm = torch.randperm(B, device=x.device)
    x_cut = x.clone()
    
    x_cut[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    
    # Calculate exact lambda
    lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / float(H * W))
    return x_cut, y, y[perm], lam_actual


# -----------------------------------------------------------------------------
# 3. ViT PATCH EXTRACTION
# -----------------------------------------------------------------------------

def extract_patches(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Unfolds Convolutional image spatial dimensions structurally into sequential flattened patches.
    Replicates `einops.rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)")` using Native torch.
    
    Args:
        imgs: Input tensor (B, C, H, W)
        patch_size: Square patch spatial length p
        
    Returns:
        Tensor of shape (B, num_patches, patch_dim) = (B, (H*W)/(p*p), p*p*C)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H},{W}) must be cleanly divisible by patch_size ({patch_size})"
        
    h_patches = H // patch_size
    w_patches = W // patch_size
    p = patch_size
    
    # View tensor recursively unrolling H -> (h_patches, p) and W -> (w_patches, p)
    x = imgs.view(B, C, h_patches, p, w_patches, p)
    # Permute to: (B, h_patches, w_patches, p, p, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    # Flatten to: (B, num_patches, spatial_element_size)
    x = x.view(B, h_patches * w_patches, p * p * C)
    
    return x


def normalize_multichannel(imgs: torch.Tensor) -> torch.Tensor:
    """
    Per-channel iterative standardisation specifically calculated individually across the batch axis 
    for Non-RGB Hyperspectral/Medical inference processing.
    """
    mean = imgs.flatten(2).mean(dim=2)                   # (B, C)
    std  = imgs.flatten(2).std(dim=2).clamp(min=1e-6)    # (B, C)
    return (imgs - mean[:, :, None, None]) / std[:, :, None, None]
