"""
ot_loss.py — Optimal Transport Style Loss for TransColor

PyTorch implementations of Sliced Wasserstein (SW), Max Sliced Wasserstein (MaxSW),
and Sinkhorn distances for replacing AdaIN-style (mean+std) loss in medical image
colorization.

Reference algorithms from hcp_reproduce/distances.py (NumPy) and gradient_flow.py

All functions are differentiable w.r.t. input features via PyTorch autograd.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ==============================================================================
# Feature conversion utilities
# ==============================================================================

def feature_to_pointcloud(
    feat: torch.Tensor,
    max_points: int = 1024,
    normalize: str = 'instance'
) -> torch.Tensor:
    """
    Convert VGG feature map (B, C, H, W) to point cloud (N_points, C).

    Args:
        feat: (B, C, H, W) feature tensor
        max_points: maximum spatial points (subsample if H*W > max_points)
        normalize: 'instance' (per-channel z-score) or 'none'

    Returns:
        (N, C) point cloud where N = B * min(H*W, max_points)
    """
    B, C, H, W = feat.shape
    N = H * W

    # Subsample if too many spatial points
    if N > max_points:
        # Adaptive average pooling to reduce spatial dimensions
        new_size = int(math.sqrt(max_points))
        feat = F.adaptive_avg_pool2d(feat, (new_size, new_size))
        B, C, H, W = feat.shape
        N = H * W

    # Reshape: (B, C, H, W) → (B, H, W, C) → (B*H*W, C)
    pointcloud = feat.permute(0, 2, 3, 1).reshape(-1, C)

    # Normalize
    if normalize == 'instance':
        # Per-channel z-score normalization across spatial points
        mean = pointcloud.mean(dim=0, keepdim=True)
        std = pointcloud.std(dim=0, keepdim=True) + 1e-8
        pointcloud = (pointcloud - mean) / std
    elif normalize == 'l2':
        pointcloud = F.normalize(pointcloud, p=2, dim=1)

    return pointcloud


def feature_list_to_pointclouds(
    feats: List[torch.Tensor],
    max_points: int = 1024,
    normalize: str = 'instance'
) -> List[torch.Tensor]:
    """Convert a list of VGG feature maps to point clouds."""
    return [feature_to_pointcloud(f, max_points, normalize) for f in feats]


# ==============================================================================
# 1D Wasserstein distance (sorting-based, closed form)
# ==============================================================================

def wasserstein_1d(x: torch.Tensor, y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Exact p-Wasserstein distance for 1D distributions via sorting.

    Args:
        x, y: 1D tensors of shape (N,) — values along projection line
        p: order of Wasserstein distance (default 2)

    Returns:
        scalar: W_p^p distance (not rooted)
    """
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    n = x_sorted.shape[0]
    diff = x_sorted - y_sorted
    if p == 2.0:
        return (diff ** 2).sum() / n
    else:
        return (diff.abs() ** p).sum() / n


# ==============================================================================
# Sliced Wasserstein Distance (1D projections)
# ==============================================================================

def sliced_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 100,
    p: float = 2.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    1D Sliced Wasserstein distance W_p^p between two point clouds.

    Projects (N, C) point clouds onto random 1D directions, computes exact
    1D Wasserstein on each projection via sorting, and averages.

    Args:
        X, Y: (N, C) point cloud tensors (N may differ between X and Y)
        n_projections: number of random 1D projection directions
        p: Wasserstein order (default 2)
        seed: optional random seed for reproducibility

    Returns:
        scalar tensor: SW_p^p distance (differentiable)
    """
    N_x, C = X.shape
    N_y = Y.shape[0]
    device = X.device

    # Generate random projection directions
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        theta = torch.randn(C, n_projections, generator=generator, device=device)
    else:
        theta = torch.randn(C, n_projections, device=device)
    theta = F.normalize(theta, p=2, dim=0)  # unit vectors, (C, L)

    # Project onto each direction: (N, C) @ (C, L) → (N, L)
    X_proj = X @ theta  # (N_x, L)
    Y_proj = Y @ theta  # (N_y, L)

    # Sort and compute 1D W_p^p per projection
    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)

    # Handle unequal N: interpolate Y to match X length if needed
    if N_x != N_y:
        # Use the longer one as reference and interpolate the shorter
        if N_y > N_x:
            # Interpolate Y_sorted down to N_x
            indices = torch.linspace(0, N_y - 1, N_x, device=device).long()
            Y_sorted = Y_sorted[indices]
        else:
            indices = torch.linspace(0, N_x - 1, N_y, device=device).long()
            X_sorted = X_sorted[indices]

    if p == 2.0:
        per_proj = (X_sorted - Y_sorted).pow(2).sum(dim=0) / N_x
    else:
        per_proj = (X_sorted - Y_sorted).abs().pow(p).sum(dim=0) / N_x

    return per_proj.mean()  # average over all projections


# ==============================================================================
# Max Sliced Wasserstein Distance
# ==============================================================================

def max_sliced_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 100,
    p: float = 2.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Max Sliced Wasserstein: take the maximum (not mean) over random projections.

    More discriminative than SW — highlights the direction of maximum distributional
    discrepancy.

    Args:
        X, Y: (N, C) point cloud tensors
        n_projections: number of random 1D projection directions
        p: Wasserstein order
        seed: optional random seed

    Returns:
        scalar tensor: MaxSW_p^p distance
    """
    N_x, C = X.shape
    N_y = Y.shape[0]
    device = X.device

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        theta = torch.randn(C, n_projections, generator=generator, device=device)
    else:
        theta = torch.randn(C, n_projections, device=device)
    theta = F.normalize(theta, p=2, dim=0)

    X_proj = X @ theta
    Y_proj = Y @ theta

    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)

    if N_x != N_y:
        if N_y > N_x:
            indices = torch.linspace(0, N_y - 1, N_x, device=device).long()
            Y_sorted = Y_sorted[indices]
        else:
            indices = torch.linspace(0, N_x - 1, N_y, device=device).long()
            X_sorted = X_sorted[indices]

    if p == 2.0:
        per_proj = (X_sorted - Y_sorted).pow(2).sum(dim=0) / N_x
    else:
        per_proj = (X_sorted - Y_sorted).abs().pow(p).sum(dim=0) / N_x

    return per_proj.max()  # max over projections


# ==============================================================================
# Sinkhorn Distance (Entropic Optimal Transport)
# ==============================================================================

def sinkhorn_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-6,
    p: float = 2.0
) -> torch.Tensor:
    """
    Entropic regularized Wasserstein distance via Sinkhorn-Knopp algorithm.

    Computes W_{p,reg} between point clouds X and Y with entropic regularization.
    Implemented in log-domain for numerical stability.

    Args:
        X, Y: (N, C) point cloud tensors (N may differ)
        reg: entropic regularization parameter (smaller = closer to exact OT)
        max_iter: maximum Sinkhorn iterations
        stop_thr: convergence threshold
        p: Wasserstein order

    Returns:
        scalar tensor: regularized W_p^p distance
    """
    N_x, C = X.shape
    N_y = Y.shape[0]
    device = X.device

    # Uniform weights
    a = torch.ones(N_x, device=device) / N_x
    b = torch.ones(N_y, device=device) / N_y

    # Cost matrix: ||x_i - y_j||^p
    C = torch.cdist(X, Y, p=2.0)
    if p == 2.0:
        C = C ** 2
    else:
        C = C ** p

    # Log-domain Sinkhorn
    # K = exp(-C / reg), but we work in log-space
    log_K = -C / reg
    log_a = torch.log(a + 1e-12)
    log_b = torch.log(b + 1e-12)

    # Initialize dual variables
    log_u = torch.zeros(N_x, device=device)
    log_v = torch.zeros(N_y, device=device)

    for _ in range(max_iter):
        log_u_prev = log_u.clone()

        # u = a / (K @ v)  →  log_u = log_a - logsumexp(log_K + log_v, dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

        # v = b / (K.T @ u)  →  log_v = log_b - logsumexp(log_K.T + log_u, dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

        # Check convergence
        if torch.max(torch.abs(log_u - log_u_prev)) < stop_thr:
            break

    # Transport plan: P = diag(exp(log_u)) @ exp(log_K) @ diag(exp(log_v))
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)

    # Wasserstein distance = sum(C * P)
    distance = torch.sum(C * P)

    return distance


# ==============================================================================
# Unified interface
# ==============================================================================

def compute_ot_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    method: str = 'SW',
    n_projections: int = 100,
    reg: float = 0.1,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Unified interface for computing OT distance between feature point clouds.

    Args:
        X, Y: (N, C) point cloud tensors
        method: 'SW' | 'MaxSW' | 'Sinkhorn'
        n_projections: number of projections for SW/MaxSW
        reg: regularization for Sinkhorn
        seed: optional random seed

    Returns:
        scalar tensor: OT distance (differentiable)
    """
    if method == 'SW':
        return sliced_wasserstein(X, Y, n_projections=n_projections, seed=seed)
    elif method == 'MaxSW':
        return max_sliced_wasserstein(X, Y, n_projections=n_projections, seed=seed)
    elif method == 'Sinkhorn':
        return sinkhorn_distance(X, Y, reg=reg)
    else:
        raise ValueError(f"Unknown OT method: {method}. Use 'SW', 'MaxSW', or 'Sinkhorn'.")


# ==============================================================================
# High-level OT Style Loss for TransColor
# ==============================================================================

class OTStyleLoss(nn.Module):
    """
    OT-based style loss that replaces AdaIN (mean+std) matching.

    Takes lists of VGG feature maps from the generated image and reference image,
    converts each to point clouds, computes OT distance per layer, and sums.

    Usage:
        ot_loss = OTStyleLoss(method='SW', n_projections=100, max_points=1024)
        loss_s = ot_loss(output_feats, style_feats)
    """

    def __init__(
        self,
        method: str = 'SW',
        n_projections: int = 100,
        max_points: int = 1024,
        normalize: str = 'instance',
        layer_weights: Optional[List[float]] = None,
        reg: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            method: 'SW', 'MaxSW', or 'Sinkhorn'
            n_projections: number of random projections (SW/MaxSW)
            max_points: max spatial points per feature map
            normalize: normalization for point clouds
            layer_weights: per-VGG-layer weights (default: uniform)
            reg: Sinkhorn regularization
            seed: random seed for reproducibility
        """
        super().__init__()
        self.method = method
        self.n_projections = n_projections
        self.max_points = max_points
        self.normalize = normalize
        self.reg = reg
        self.seed = seed

        if layer_weights is None:
            # Default: weight early (color-relevant) layers more
            self.layer_weights = [1.0, 1.0, 0.5, 0.25, 0.125]
        else:
            self.layer_weights = layer_weights

    def forward(
        self,
        output_feats: List[torch.Tensor],
        style_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            output_feats: list of (B, C, H, W) from generated image Ics
            style_feats: list of (B, C, H, W) from reference image Is

        Returns:
            scalar OT style loss
        """
        total_loss = 0.0
        n_layers = min(len(output_feats), len(style_feats), len(self.layer_weights))

        for i in range(n_layers):
            # Convert feature maps to point clouds
            x = feature_to_pointcloud(
                output_feats[i], max_points=self.max_points,
                normalize=self.normalize
            )
            y = feature_to_pointcloud(
                style_feats[i], max_points=self.max_points,
                normalize=self.normalize
            )

            # Compute OT distance for this layer
            layer_dist = compute_ot_distance(
                x, y,
                method=self.method,
                n_projections=self.n_projections,
                reg=self.reg,
                seed=self.seed
            )

            total_loss += self.layer_weights[i] * layer_dist

        return total_loss
