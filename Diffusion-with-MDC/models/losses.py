"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th

import torch.nn.functional as F


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
'''
def low_high_frequency_loss(x, y, cutoff_freq=0.3):
    """
    Compute weighted loss for low and high frequency components
    Args:
        x: Generated image [B, C, H, W]
        y: Reference image [B, C, H, W]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_low, loss_high = 0, 0
    eps = 1e-8
    for c in range(C):
        x_freq = th.fft.fft2(x[:,c])
        y_freq = th.fft.fft2(y[:,c])
        x_freq = th.fft.fftshift(x_freq)
        y_freq = th.fft.fftshift(y_freq)
        mask = th.ones((H, W), device=x.device)
        center_h, center_w = H // 2, W // 2
        freq_radius = int(min(H, W) * cutoff_freq / 2)
        y_coords, x_coords = th.meshgrid(th.arange(H, device=x.device), th.arange(W, device=x.device))
        dist_from_center = th.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        low_mask = (dist_from_center <= freq_radius).float()
        high_mask = 1.0 - low_mask
        # Low-frequency loss
        x_freq_low = x_freq * low_mask
        y_freq_low = y_freq * low_mask
        # Apply log to magnitude then compute MSE
        loss_low += F.mse_loss(th.log(x_freq_low.abs() + eps), th.log(y_freq_low.abs() + eps))
        # loss_low += F.mse_loss(x_freq_low.angle(), y_freq_low.angle())
        # High-frequency loss
        x_freq_high = x_freq * high_mask
        y_freq_high = y_freq * high_mask
        # Apply log to magnitude then compute MSE
        loss_high += F.mse_loss(th.log(x_freq_high.abs() + eps), th.log(y_freq_high.abs() + eps))
        # loss_high += F.mse_loss(x_freq_high.angle(), y_freq_high.angle())
    return 0.5 * (loss_low / C) + 0.5 * (loss_high / C)



### 2025-05-28 Update: compute MSE directly without log
def low_high_frequency_loss(x, y, cutoff_freq=0.3):
    """
    Compute weighted loss for low and high frequency components
    Args:
        x: Generated image [B, C, H, W]
        y: Reference image [B, C, H, W]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_low, loss_high = 0, 0
    eps = 1e-8
    for c in range(C):
        x_freq = th.fft.fft2(x[:,c])
        y_freq = th.fft.fft2(y[:,c])
        x_freq = th.fft.fftshift(x_freq)
        y_freq = th.fft.fftshift(y_freq)
        mask = th.ones((H, W), device=x.device)
        center_h, center_w = H // 2, W // 2
        freq_radius = int(min(H, W) * cutoff_freq / 2)
        y_coords, x_coords = th.meshgrid(th.arange(H, device=x.device), th.arange(W, device=x.device))
        dist_from_center = th.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        low_mask = (dist_from_center <= freq_radius).float()
        high_mask = 1.0 - low_mask
        # Low-frequency loss
        x_freq_low = x_freq * low_mask
        y_freq_low = y_freq * low_mask
        # Normalize data
        x_freq_low = x_freq_low / (x_freq_low.abs() + eps)
        y_freq_low = y_freq_low / (y_freq_low.abs() + eps)
        loss_low += F.mse_loss(x_freq_low.abs(), y_freq_low.abs())
        # loss_low += F.mse_loss(x_freq_low.angle(), y_freq_low.angle())
        # High-frequency loss
        x_freq_high = x_freq * high_mask
        y_freq_high = y_freq * high_mask
        # Normalize data
        x_freq_high = x_freq_high / (x_freq_high.abs() + eps)
        y_freq_high = y_freq_high / (y_freq_high.abs() + eps)
        loss_high += F.mse_loss(x_freq_high.abs(), y_freq_high.abs())
        # loss_high += F.mse_loss(x_freq_high.angle(), y_freq_high.angle())
    return 0.5 * (loss_low) + 0.5 * (loss_high)

'''

### 2025-05-29 Update: add Gaussian smoothing
def low_high_frequency_loss(x, y, cutoff_freq=0.3):
    """
    Improved frequency loss function
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_low, loss_high = 0, 0
    eps = 1e-8
    
    for c in range(C):
        # 1. Add Gaussian smoothing preprocessing
        x_smooth = F.avg_pool2d(x[:,c:c+1], kernel_size=3, stride=1, padding=1)
        y_smooth = F.avg_pool2d(y[:,c:c+1], kernel_size=3, stride=1, padding=1)
        
        # 2. FFT transform
        x_freq = th.fft.fft2(x_smooth.squeeze(1))
        y_freq = th.fft.fft2(y_smooth.squeeze(1))
        x_freq = th.fft.fftshift(x_freq)
        y_freq = th.fft.fftshift(y_freq)
        
        # 3. Use smoother mask
        center_h, center_w = H // 2, W // 2
        y_coords, x_coords = th.meshgrid(th.arange(H, device=x.device), th.arange(W, device=x.device))
        dist_from_center = th.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        freq_radius = int(min(H, W) * cutoff_freq / 2)
        
        # Use Gaussian mask instead of hard threshold
        low_mask = th.exp(-dist_from_center**2 / (2 * (freq_radius/2)**2))
        high_mask = 1.0 - low_mask
        
        # 4. Normalize frequency domain coefficients
        x_freq_norm = x_freq / (x_freq.abs().max() + eps)
        y_freq_norm = y_freq / (y_freq.abs().max() + eps)
        
        # 5. Use L1 loss instead of MSE
        loss_low += F.l1_loss(x_freq_norm * low_mask, y_freq_norm * low_mask)
        loss_high += F.l1_loss(x_freq_norm * high_mask, y_freq_norm * high_mask)
    
    return 0.5 * (loss_low / C) + 0.5 * (loss_high / C)

'''

def qr_structure_detail_loss(x, y, weight_q=0.5, weight_r=0.5):
    """
    Apply QR decomposition to inputs x and y, compute loss for Q (structure) and R (detail) separately
    Args:
        x: Generated image [B, C, H, W]
        y: Reference image [B, C, H, W]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_q, loss_r = 0, 0
    for b in range(B):
        for c in range(C):
            # Extract single-channel 2D image
            x_mat = x[b, c]
            y_mat = y[b, c]
            # QR decomposition
            Qx, Rx = th.linalg.qr(x_mat)
            Qy, Ry = th.linalg.qr(y_mat)
            # Structure loss
            loss_q += F.mse_loss(Qx, Qy)
            # Detail loss
            # loss_r += F.mse_loss(Rx, Ry)
            # Detail loss using L1 loss
            loss_r += F.l1_loss(Rx, Ry)
    return weight_q * loss_q / (B * C) + weight_r * loss_r / (B * C)

### 2025-05-28 QR loss normalization
def qr_structure_detail_loss(x, y, weight_q=0.2, weight_r=0.8):
    """
    Apply QR decomposition to inputs x and y, compute loss for Q (structure) and R (detail) separately
    Args:
        x: Generated image [B, C, H, W]
        y: Reference image [B, C, H, W]
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_q, loss_r = 0, 0
    eps = 1e-8
    for b in range(B):
        for c in range(C):
            # Extract single-channel 2D image
            x_mat = x[b, c]
            y_mat = y[b, c]
            # QR decomposition
            Qx, Rx = th.linalg.qr(x_mat)
            Qy, Ry = th.linalg.qr(y_mat)
            # Normalize data
            Qx = Qx / (Qx.abs() + eps)
            Qy = Qy / (Qy.abs() + eps)
            Rx = Rx / (Rx.abs() + eps)
            Ry = Ry / (Ry.abs() + eps)
            # Structure loss
            loss_q += F.mse_loss(Qx, Qy)
            # Detail loss
            # loss_r += F.mse_loss(Rx, Ry)
            # Detail loss using L1 loss
            loss_r += F.l1_loss(Rx, Ry)
    return weight_q * loss_q + weight_r * loss_r 
'''
def qr_structure_detail_loss(x, y, weight_q=0.8, weight_r=0.2):
    """
    Improved QR structure loss function
    """
    assert x.shape == y.shape
    B, C, H, W = x.shape
    loss_q, loss_r = 0, 0
    eps = 1e-8
    
    for b in range(B):
        for c in range(C):
            # 1. Preprocessing: add Gaussian smoothing
            x_mat = F.avg_pool2d(x[b:b+1,c:c+1], kernel_size=3, stride=1, padding=1).squeeze()
            y_mat = F.avg_pool2d(y[b:b+1,c:c+1], kernel_size=3, stride=1, padding=1).squeeze()
            
            # 2. Normalize
            x_mat = x_mat / (x_mat.abs().max() + eps)
            y_mat = y_mat / (y_mat.abs().max() + eps)
            
            # 3. QR decomposition
            Qx, Rx = th.linalg.qr(x_mat)
            Qy, Ry = th.linalg.qr(y_mat)
            
            # 4. Normalize Q and R
            Qx = Qx / (Qx.abs().max() + eps)
            Qy = Qy / (Qy.abs().max() + eps)
            Rx = Rx / (Rx.abs().max() + eps)
            Ry = Ry / (Ry.abs().max() + eps)
            
            # 5. Use L1 loss
            loss_q += F.l1_loss(Qx, Qy)
            loss_r += F.l1_loss(Rx, Ry)
    
    return weight_q * loss_q / (B * C) + weight_r * loss_r / (B * C)
