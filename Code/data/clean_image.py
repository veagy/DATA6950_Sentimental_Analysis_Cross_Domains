import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

def denoise_image(image_path_or_array, method: str = 'bilateral') -> np.ndarray:
    """
    Denoise an image using OpenCV.
    method: 'gaussian', 'median', 'bilateral', 'nlm'
    """
    if cv2 is None:
        raise ImportError("Please install opencv-python: `pip install opencv-python` to use image denoising.")

    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array
    else:
        raise ValueError("Must provide either a path string or a numpy array")

    method = method.lower()
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
    elif method == 'median':
        return cv2.medianBlur(img, ksize=3)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'nlm':
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
        else:
            return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def median_filter_tensor(img: torch.Tensor, size: int = 3) -> torch.Tensor:
    """Apply median filter to a (C, H, W) float32 tensor."""
    if ndimage is None:
        raise ImportError("Please install scipy: `pip install scipy` to use median_filter_tensor.")

    assert img.dim() == 3, "Input tensor must be 3D (C, H, W)"
    
    img_np = img.permute(1, 2, 0).numpy()   # → (H, W, C)
    filtered = np.stack([ndimage.median_filter(img_np[:,:,c], size=size)
                         for c in range(img_np.shape[2])], axis=2)
    return torch.from_numpy(filtered).permute(2, 0, 1)
