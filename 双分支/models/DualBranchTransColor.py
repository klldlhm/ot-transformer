"""
DualBranchTransColor.py — Dual-branch architecture for OT-TransColor

Architecture:
- Color Branch (CNN + OT): Learn color distribution mapping with Optimal Transport
- Structure Branch (Transformer): Preserve anatomical structure with self-attention
- Fusion Module: Combine color and structure outputs

Key insight:
- CNN + OT loss naturally aligns color distributions
- Transformer + content loss preserves long-range structural dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from function import calc_mean_std, normal


class ColorBranch(nn.Module):
    """
    Lightweight CNN for color distribution learning.
    Focus on global color statistics, not fine-grained spatial details.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Encoder: extract color features at reduced resolution
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((28, 28))  # Fixed size for color transfer
        )

        # Decoder: reconstruct colored image
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            color_feat: (B, 256, 28, 28) color features
            color_output: (B, 3, H, W) colored image
        """
        feat = self.encoder(x)
        out = self.decoder(feat)
        return feat, out


class StructureBranch(nn.Module):
    """
    Transformer-based structure preservation branch.
    Uses self-attention to maintain long-range spatial relationships.
    """
    def __init__(self, patch_embed, transformer, decoder):
        super().__init__()
        self.patch_embed = patch_embed
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, content, style=None):
        """
        Args:
            content: (B, 3, H, W) content image
            style: (B, 3, H, W) style image (optional, for cross-attention)
        Returns:
            struct_output: (B, 3, H, W) structure-preserved output
            content_feat: (B, C, H', W') transformer features of content
        """
        # Patch embedding
        content_feat = self.patch_embed(content)

        if style is not None:
            style_feat = self.patch_embed(style)
            # Cross-attention: query from content, key/value from style
            trans_out = self.transformer(style_feat, None, content_feat)
        else:
            # Self-attention: preserve content structure
            trans_out = self.transformer(content_feat, None, content_feat)

        struct_output = self.decoder(trans_out)

        return struct_output, content_feat


class FusionModule(nn.Module):
    """
    Adaptive fusion of color and structure branches.
    Learn spatially-varying weights to combine both outputs.
    """
    def __init__(self, in_channels=6, hidden_channels=64):
        super().__init__()

        self.fusion_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        # Attention weights for adaptive fusion
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),  # 2 channels: weight for color vs structure
            nn.Softmax(dim=1)
        )

    def forward(self, color_output, struct_output):
        """
        Args:
            color_output: (B, 3, H, W) from color branch
            struct_output: (B, 3, H, W) from structure branch
        Returns:
            fused: (B, 3, H, W) fused output
        """
        # Concatenate both outputs
        combined = torch.cat([color_output, struct_output], dim=1)

        # Residual fusion
        fused = self.fusion_net(combined)

        # Attention-weighted fusion
        weights = self.attention(combined)  # (B, 2, H, W)
        w_color = weights[:, 0:1, :, :]
        w_struct = weights[:, 1:2, :, :]

        # Combine: fused + weighted average
        adaptive_fused = w_color * color_output + w_struct * struct_output

        return fused + 0.5 * adaptive_fused


class DualBranchColorTrans(nn.Module):
    """
    Dual-branch architecture for medical image colorization.

    Key design choices:
    1. Color Branch (CNN): Lightweight, focuses on global color distribution
       - Trained with OT loss for complete distribution matching
    2. Structure Branch (Transformer): Preserves anatomical structure
       - Trained with content loss for structural consistency
    3. Fusion Module: Learns to combine color and structure
       - Spatially-adaptive weighting
    """
    def __init__(self, vgg_encoder, decoder, patch_embed, transformer, args):
        super().__init__()

        # Frozen VGG for perceptual loss computation
        enc_layers = list(vgg_encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # === Three branches ===
        self.color_branch = ColorBranch(in_channels=3, base_channels=64)
        self.structure_branch = StructureBranch(patch_embed, transformer, decoder)
        self.fusion_module = FusionModule(in_channels=6, hidden_channels=64)

        # Loss modules
        self.mse_loss = nn.MSELoss()
        self.ot_loss_module = None

        # Loss weights (can be tuned)
        self.color_weight = getattr(args, 'color_weight', 10.0)
        self.structure_weight = getattr(args, 'structure_weight', 7.0)
        self.fusion_weight = getattr(args, 'fusion_weight', 5.0)

    def set_ot_style_loss(self, method='SW', n_projections=100, max_points=1024,
                          normalize='instance', reg=0.1):
        """Configure OT loss for color branch."""
        from ot_loss import OTStyleLoss
        self.ot_loss_module = OTStyleLoss(
            method=method,
            n_projections=n_projections,
            max_points=max_points,
            normalize=normalize,
            reg=reg,
        )
        print(f"[DualBranch] OT loss configured: {method}")

    def encode_with_intermediate(self, input):
        """Extract VGG features at 5 layers."""
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        """MSE content loss."""
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_ot_style_loss(self, input_feats, style_feats):
        """OT-based style loss."""
        if self.ot_loss_module is None:
            raise RuntimeError("OT loss not configured. Call set_ot_style_loss() first.")
        return self.ot_loss_module(input_feats, style_feats)

    def forward(self, samples_c, samples_s, ot_method='SW', ot_n_proj=100):
        """
        Forward pass with dual branches.

        Args:
            samples_c: content images (grayscale medical images)
            samples_s: style images (real color tissue slices)
            ot_method: OT method for color loss
            ot_n_proj: number of projections for SW/MaxSW

        Returns:
            fused: final colorized output
            loss_color: OT loss on color branch
            loss_structure: content loss on structure branch
            loss_fusion: perceptual loss on fused output
            color_output: intermediate color branch output
            struct_output: intermediate structure branch output
        """
        # Handle NestedTensor conversion
        if isinstance(samples_c, (list, torch.Tensor)):
            if isinstance(samples_c, list):
                content_input = torch.stack(samples_c) if len(samples_c) > 1 else samples_c[0].unsqueeze(0)
            else:
                content_input = samples_c
            samples_c_nested = nested_tensor_from_tensor_list(
                [content_input] if content_input.dim() == 3 else content_input
            )
        else:
            samples_c_nested = samples_c
            content_input = samples_c.tensors

        if isinstance(samples_s, (list, torch.Tensor)):
            if isinstance(samples_s, list):
                style_input = torch.stack(samples_s) if len(samples_s) > 1 else samples_s[0].unsqueeze(0)
            else:
                style_input = samples_s
            samples_s_nested = nested_tensor_from_tensor_list(
                [style_input] if style_input.dim() == 3 else style_input
            )
        else:
            samples_s_nested = samples_s
            style_input = samples_s.tensors

        # === Forward through branches ===

        # 1. Color Branch: extract color distribution from style, apply to content
        content_color_feat, color_output = self.color_branch(content_input)
        style_color_feat, _ = self.color_branch(style_input)

        # 2. Structure Branch: preserve content structure
        struct_output, content_trans_feat = self.structure_branch(
            content_input, style_input
        )

        # 3. Fusion: combine color and structure
        fused = self.fusion_module(color_output, struct_output)

        # === Loss Computation ===

        # VGG features for all outputs
        color_feats = self.encode_with_intermediate(color_output)
        struct_feats = self.encode_with_intermediate(struct_output)
        fused_feats = self.encode_with_intermediate(fused)
        style_feats = self.encode_with_intermediate(style_input)
        content_feats = self.encode_with_intermediate(content_input)

        # 1. Color Loss: OT distance on color branch output
        if ot_method != 'none':
            if self.ot_loss_module is None:
                self.set_ot_style_loss(method=ot_method, n_projections=ot_n_proj)
            loss_color_ot = self.calc_ot_style_loss(color_feats, style_feats)
        else:
            loss_color_ot = torch.tensor(0.0, device=content_input.device)

        # 2. Structure Loss: content preservation on structure branch
        loss_structure = self.calc_content_loss(
            normal(struct_feats[-1]), normal(content_feats[-1])
        ) + self.calc_content_loss(
            normal(struct_feats[-2]), normal(content_feats[-2])
        )

        # 3. Fusion Loss: perceptual quality of final output
        loss_fusion_content = self.calc_content_loss(
            normal(fused_feats[-1]), normal(content_feats[-1])
        )
        loss_fusion_style = self.calc_ot_style_loss(fused_feats, style_feats) if ot_method != 'none' else torch.tensor(0.0)
        loss_fusion = loss_fusion_content + 0.5 * loss_fusion_style

        # 4. Identity losses (regularization)
        # Color branch identity: Ic -> Ic should remain unchanged
        _, color_identity = self.color_branch(content_input)
        loss_color_identity = self.calc_content_loss(color_identity, content_input)

        # Structure branch identity: Ic -> Ic should remain unchanged
        struct_identity, _ = self.structure_branch(content_input, content_input)
        loss_struct_identity = self.calc_content_loss(struct_identity, content_input)

        loss_identity = loss_color_identity + loss_struct_identity

        return (fused, loss_color_ot, loss_structure, loss_fusion,
                loss_identity, color_output, struct_output)
