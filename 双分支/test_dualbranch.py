"""
test_dualbranch.py — Inference script for Dual-Branch OT-TransColor

Generates outputs from all three components:
  - Color branch only
  - Structure branch only
  - Fused output

Usage:
    python test_dualbranch.py \
        --content ./content_samples \
        --style ./style_samples \
        --output ./results/dualbranch \
        --checkpoint experiments/dualbranch/dualbranch_iter_160000.pth \
        --ot_method SW
"""
import argparse
import os
import torch
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # 项目根（dual/ 的上一层）
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.transformer import Transformer
from models.DualBranchTransColor import DualBranchColorTrans
from models.ViT_helper import to_2tuple
import torch.nn as nn
from models.TransColor import vgg, decoder
from util.misc import nested_tensor_from_tensor_list


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=512):
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


def get_args():
    parser = argparse.ArgumentParser(description='Dual-Branch Testing')

    # === Paths ===
    parser.add_argument('--content', type=str, required=True,
                        help='Content image or directory')
    parser.add_argument('--style', type=str, required=True,
                        help='Style image or directory')
    parser.add_argument('--output', type=str, default=str(ROOT / 'results' / 'dualbranch'),
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to dual-branch checkpoint')

    # === Pre-trained modules ===
    parser.add_argument('--vgg', type=str, default=str(ROOT / 'experiments' / 'vgg_normalised.pth'))
    parser.add_argument('--decoder', type=str, default=str(ROOT / 'experiments' / 'decoder_iter_160000.pth'))
    parser.add_argument('--Trans', type=str, default=str(ROOT / 'experiments' / 'transformer_iter_160000.pth'))
    parser.add_argument('--embedding', type=str, default=str(ROOT / 'experiments' / 'embedding_iter_160000.pth'))

    # === OT Configuration ===
    parser.add_argument('--ot_method', type=str, default='SW',
                        choices=['none', 'SW', 'MaxSW', 'Sinkhorn'])
    parser.add_argument('--ot_n_projections', type=int, default=100)
    parser.add_argument('--ot_max_points', type=int, default=1024)

    # === Testing ===
    parser.add_argument('--save_components', action='store_true',
                        help='Save color/structure branches separately')

    args = parser.parse_args()
    return args


def load_image(image_path, device):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def main():
    args = get_args()

    # === Device ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DualBranch Test] Using device: {device}")

    # === Output directory ===
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load pre-trained modules ===
    embedding = PatchEmbed()
    Trans = Transformer()

    decoder.load_state_dict(torch.load(args.decoder))
    embedding.load_state_dict(torch.load(args.embedding))
    Trans.load_state_dict(torch.load(args.Trans))
    vgg.load_state_dict(torch.load(args.vgg))

    decoder.eval()
    vgg.eval()

    # === Create network ===
    class DummyArgs:
        color_weight = 10.0
        structure_weight = 7.0
        fusion_weight = 5.0

    network = DualBranchColorTrans(vgg, decoder, embedding, Trans, DummyArgs()).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    # Configure OT
    if args.ot_method != 'none':
        network.set_ot_style_loss(
            method=args.ot_method,
            n_projections=args.ot_n_projections,
            max_points=args.ot_max_points,
        )

    print(f"[DualBranch Test] Loaded checkpoint from iteration {checkpoint['iteration']}")

    # === Get image pairs ===
    content_path = Path(args.content)
    style_path = Path(args.style)

    if content_path.is_dir():
        content_images = sorted(content_path.glob('*.jpg')) + sorted(content_path.glob('*.png'))
    else:
        content_images = [content_path]

    if style_path.is_dir():
        style_images = sorted(style_path.glob('*.jpg')) + sorted(style_path.glob('*.png'))
    else:
        style_images = [style_path]

    print(f"[DualBranch Test] Found {len(content_images)} content images, {len(style_images)} style images")

    # === Inference ===
    with torch.no_grad():
        for i, content_img_path in enumerate(content_images):
            for j, style_img_path in enumerate(style_images):
                print(f"Processing pair ({i+1}/{len(content_images)}, {j+1}/{len(style_images)})")

                # Load images
                content_img = load_image(content_img_path, device)
                style_img = load_image(style_img_path, device)

                # Convert to NestedTensor
                samples_c = nested_tensor_from_tensor_list(content_img)
                samples_s = nested_tensor_from_tensor_list(style_img)

                # Forward
                (fused, loss_color_ot, loss_structure, loss_fusion,
                 loss_identity, color_output, struct_output) = network(
                    samples_c, samples_s,
                    ot_method=args.ot_method,
                    ot_n_proj=args.ot_n_projections
                )

                # Save outputs
                content_name = content_img_path.stem
                style_name = style_img_path.stem
                prefix = f"{content_name}_stylized_by_{style_name}"

                # Main fused output
                save_image(fused, output_dir / f"{prefix}_fused.jpg", normalize=True)

                # Optional: save individual branches
                if args.save_components:
                    save_image(color_output, output_dir / f"{prefix}_color_branch.jpg", normalize=True)
                    save_image(struct_output, output_dir / f"{prefix}_structure_branch.jpg", normalize=True)

                    # Comparison grid
                    comparison = torch.cat([
                        content_img,
                        style_img,
                        color_output,
                        struct_output,
                        fused
                    ], dim=0)
                    save_image(comparison, output_dir / f"{prefix}_comparison.jpg",
                              nrow=5, normalize=True)

    print(f"[DualBranch Test] Complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
