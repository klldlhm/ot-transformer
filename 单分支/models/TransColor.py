import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from function import normal
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ColorTrans(nn.Module):
    """ This is the style transform transformer module """

    def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.decode = decoder
        self.embedding = PatchEmbed

        # OT style loss configuration (set during training, None = use AdaIN)
        self.ot_style_loss_module = None

    def set_ot_style_loss(self, method='SW', n_projections=100, max_points=1024):
        """Configure OT-based style loss. Call before training."""
        from ot_loss import OTStyleLoss
        self.ot_style_loss_module = OTStyleLoss(
            method=method,
            n_projections=n_projections,
            max_points=max_points,
            normalize='instance'
        )
        print(f"[OT-TransColor] Style loss: {method} (n_proj={n_projections}, max_pts={max_points})")

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        """Original AdaIN style loss: MSE on mean + MSE on std (2nd moment only)"""
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_ot_style_loss(self, input_feats, style_feats):
        """
        OT-based style loss: complete distribution matching via Optimal Transport.
        Replaces AdaIN (mean+std) matching with full distribution matching.

        Args:
            input_feats: list of (B,C,H,W) from encode_with_intermediate(Ics)
            style_feats: list of (B,C,H,W) from encode_with_intermediate(style_img)
        Returns:
            scalar OT style loss
        """
        if self.ot_style_loss_module is None:
            raise RuntimeError("OT style loss not configured. Call set_ot_style_loss() first.")
        return self.ot_style_loss_module(input_feats, style_feats)

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor,
                ot_method='SW', ot_n_proj=100):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Args:
            samples_c: content images (grey-scale medical)
            samples_s: style images (real human color slices)
            ot_method: 'SW' | 'MaxSW' | 'Sinkhorn' | 'none' (none = AdaIN only)
            ot_n_proj: number of projections for SW/MaxSW
        """
        content_input = samples_c
        style_input = samples_s

        # Save tensor form for identity loss (before conversion to NestedTensor)
        if isinstance(samples_c, torch.Tensor):
            content_input = samples_c
        elif isinstance(samples_c, list):
            content_input = torch.stack(samples_c) if len(samples_c) > 1 else samples_c[0].unsqueeze(0)

        if isinstance(samples_s, torch.Tensor):
            style_input = samples_s
        elif isinstance(samples_s, list):
            style_input = torch.stack(samples_s) if len(samples_s) > 1 else samples_s[0].unsqueeze(0)

        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        # features used to calculate loss
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        # Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        mask = None
        hs = self.transformer(style, mask, content)
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)

        # ---- Content Loss (unchanged) ----
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                 self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))

        # ---- Style Loss: AdaIN (always computed for monitoring) ----
        loss_s_adain = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s_adain += self.calc_style_loss(Ics_feats[i], style_feats[i])

        # ---- Style Loss: OT (computed when ot_method != 'none') ----
        loss_s_ot = torch.tensor(0.0, device=loss_s_adain.device)
        if ot_method != 'none':
            # Configure OT module if needed
            if self.ot_style_loss_module is None:
                self.set_ot_style_loss(method=ot_method, n_projections=ot_n_proj)
            loss_s_ot = self.calc_ot_style_loss(Ics_feats, style_feats)

        # ---- Identity Losses (unchanged) ----
        Icc = self.decode(self.transformer(content, mask, content))
        Iss = self.decode(self.transformer(style, mask, style))

        # Identity losses lambda 1 (pixel-level)
        loss_lambda1 = self.calc_content_loss(Icc, content_input) + \
                       self.calc_content_loss(Iss, style_input)

        # Identity losses lambda 2 (feature-level)
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + \
                       self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + \
                            self.calc_content_loss(Iss_feats[i], style_feats[i])

        # Return: image, content_loss, style_loss_adain, style_loss_ot, id1, id2
        return Ics, loss_c, loss_s_adain, loss_s_ot, loss_lambda1, loss_lambda2
