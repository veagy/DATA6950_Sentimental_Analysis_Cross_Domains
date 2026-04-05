"""
Phase 4: Modality-Specific Video Preprocessing.
Supports PyTorch temporal sequencing, clipping, and RAFT Optical flow mathematics.
"""

import warnings
import torch
from torch.utils.data import Dataset

try:
    import torchvision.io as tvio
except Exception:
    tvio = None

try:
    from torchvision.transforms import v2 as T
except Exception:
    T = None


# -----------------------------------------------------------------------------
# 1. VIDEO LOADING & TEMPORAL SAMPLING
# -----------------------------------------------------------------------------

def load_video_tensor(file_path: str, start_pts: float = 0.0, end_pts: float = 10.0, fps: int = -1):
    """
    Unpacks encoded videos statically to normalised sequential frames.
    Returns:
        Float tensor structure of shape (C, T, H, W) in domain [0, 1]
    """
    if tvio is None:
        raise ImportError("pip install torchvision required for Video decoding.")
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        video, audio, info = tvio.read_video(
            file_path,
            pts_unit="sec",
            start_pts=start_pts,
            end_pts=end_pts
        )
        
    # Structure native: (T, H, W, C) -> Reproject to PyTorch (C, T, H, W)
    video = video.permute(3, 0, 1, 2).contiguous().float() / 255.0
    return video, info


def spatial_video_transform(video: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Applies unified sequential image crops structurally across the temporal sequence T mapping to standard sizes.
    Args:
        video: (C, T, H, W)
    """
    if T is None:
        raise ImportError("torchvision transforms.v2 required.")
        
    resize_dim = int(image_size / 0.875)
    
    transform_block = T.Compose([
        T.Resize(resize_dim, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    # Process temporally iteratively
    processed = torch.stack([
        transform_block(video[:, t, :, :]) for t in range(video.shape[1])
    ], dim=1) # Restored shape: (C, T, H_new, W_new)
    
    return processed


# -----------------------------------------------------------------------------
# 2. DENSE OPTICAL FLOW NATIVE
# -----------------------------------------------------------------------------

def compute_dense_optical_flow(frames: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Computes rigorous temporal RAFT optical displacements predicting the dense transition parameters natively.
    Args:
        frames: sequence structural representation (C, T, H, W) in float range [0, 1]
    Returns:
        (2, T-1, H, W) flow matrix representing mapping coordinates per-frame sequence mathematically.
    """
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except ImportError:
        raise ImportError("torchvision > 0.12 required for RAFT models.")
        
    weights = Raft_Large_Weights.DEFAULT
    preprocess_r = weights.transforms()
    
    raft = raft_large(weights=weights).to(device)
    raft.eval()
    
    # Reproject (C, T, H, W) to (T, C, H, W) for pair-matching sequence extraction
    frames_t = frames.permute(1, 0, 2, 3).to(device)
    
    flows = []
    with torch.no_grad():
        for t in range(frames_t.shape[0] - 1):
            f1 = frames_t[t:t+1]
            f2 = frames_t[t+1:t+2]
            
            f1_p, f2_p = preprocess_r(f1, f2)
            predicted_flows = raft(f1_p, f2_p)
            
            # Predicts structural tuples, index highest accuracy flow at prediction block length -1
            flows.append(predicted_flows[-1].cpu())
            
    # Matrix output returns to contiguous batch: (T-1, 2, H, W)
    flow_tensor = torch.cat(flows, dim=0)
    
    # Standardise return format mapping to (Channels, Time, H, W) -> (2, T-1, H, W)
    return flow_tensor.permute(1, 0, 2, 3)


# -----------------------------------------------------------------------------
# 3. TEMPORAL CLIP SLICING AND FIXED BATCHING PADS
# -----------------------------------------------------------------------------

def build_clips(video: torch.Tensor, clip_frames: int = 16, stride: int = 8) -> list:
    """
    Segments raw lengthy sequence vectors identically separating spatial ranges via overlapped boundaries.
    Args:
        video: (C, T, H, W) tensor block.
        clip_frames: Length of bounding cut sizes.
        stride: Offset spacing window parameters (<= clip_frames for boundary prediction padding loops).
    """
    T_total = video.shape[1]
    clips = []
    for start in range(0, T_total, stride):
        end = min(start + clip_frames, T_total)
        
        chunk = video[:, start:end, :, :]
        # Pad temporal chunk length sequence if final remainder is inadequate length padding zeros implicitly.
        if chunk.shape[1] < clip_frames:
            num_pads = clip_frames - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, num_pads))
            
        clips.append(chunk)
        
    return clips


class VideoClipDataset(Dataset):
    """
    PyTorch memory-efficient frame pointer dataset loader that extracts temporal chunks utilizing clip slicing iteratively natively directly from encoded MP4 block logic.
    """
    def __init__(self, video_paths: list, labels: list, clip_frames: int = 16, stride: int = 8, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.clip_frames = clip_frames
        self.stride = stride
        self.transform = transform
        
        self.index = []
        for v_idx, path in enumerate(video_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    video, _, _ = tvio.read_video(path, pts_unit="sec")
                    T_total = video.shape[0]
                    for start in range(0, T_total, stride):
                        self.index.append((v_idx, start))
                except Exception:
                    pass
                    
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):
        v_idx, start = self.index[idx]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video, _, _ = tvio.read_video(self.video_paths[v_idx], pts_unit="sec")
            
        # (T, H, W, C)
        clip = video[start:start + self.clip_frames]
        # Restructure to normative math constraint mappings -> (C, T, H, W) Domain [0,1]
        clip = clip.permute(3, 0, 1, 2).contiguous().float() / 255.0
        
        if clip.shape[1] < self.clip_frames:
            pad_size = self.clip_frames - clip.shape[1]
            clip = torch.nn.functional.pad(clip, (0, 0, 0, 0, 0, pad_size))
            
        if self.transform:
            clip = torch.stack([self.transform(clip[:, t]) for t in range(clip.shape[1])], dim=1)
            
        return clip, self.labels[v_idx]
