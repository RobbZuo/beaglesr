import torch
import torch.fft as fft


def kspace_truncate_2d(img: torch.Tensor, sf: int = 4) -> torch.Tensor:
    """
    img: (B, C, H, W), real-valued, roughly in [0, 1]
    return: (B, C, H//sf, W//sf), real-valued
    Keep the central 1/sf × 1/sf of k-space, IFFT → LR.
    """
    assert img.ndim == 4
    b, c, h, w = img.shape
    assert h % sf == 0 and w % sf == 0, f"H,W must be divisible by sf={sf}, got {h},{w}"

    # per-channel FFT
    k = fft.fftshift(fft.fft2(img, norm="ortho"), dim=(-2, -1))

    h_lr, w_lr = h // sf, w // sf
    y0 = (h - h_lr) // 2
    x0 = (w - w_lr) // 2
    k_lr = k[:, :, y0:y0 + h_lr, x0:x0 + w_lr]

    lr = fft.ifft2(fft.ifftshift(k_lr, dim=(-2, -1)), norm="ortho").real
    return lr.clamp(0.0, 1.0)