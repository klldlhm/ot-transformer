"""
train_dualbranch.py — Training script for Dual-Branch OT-TransColor

Usage:
    python train_dualbranch.py \
        --content_dir ./datasets/train2014 \
        --style_dir ./datasets/PH2 \
        --ot_method SW \
        --ot_n_projections 100 \
        --color_weight 10.0 \
        --structure_weight 7.0 \
        --fusion_weight 5.0
"""
import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # 项目根（dual/ 的上一层）
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from sampler import InfiniteSamplerWrapper
from models.transformer import Transformer
from models.DualBranchTransColor import DualBranchColorTrans
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from models.TransColor import vgg, decoder
from util.misc import nested_tensor_from_tensor_list
from PIL import Image
from fair_experiment import (
    assert_finite,
    build_run_metadata,
    set_seed,
    write_run_metadata,
)


def get_args():
    parser = argparse.ArgumentParser(description='Dual-Branch OT-TransColor Training')

    # === Paths ===
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Path to content images (e.g., grayscale medical images)')
    parser.add_argument('--style_dir', type=str, required=True,
                        help='Path to style images (e.g., color tissue slices)')
    parser.add_argument('--vgg', type=str, default=str(ROOT / 'experiments' / 'vgg_normalised.pth'),
                        help='Path to pre-trained VGG')
    parser.add_argument('--decoder', type=str, default=str(ROOT / 'experiments' / 'decoder_iter_160000.pth'),
                        help='Path to pre-trained decoder')
    parser.add_argument('--Trans', type=str, default=str(ROOT / 'experiments' / 'transformer_iter_160000.pth'),
                        help='Path to pre-trained Transformer')
    parser.add_argument('--embedding', type=str, default=str(ROOT / 'experiments' / 'embedding_iter_160000.pth'),
                        help='Path to pre-trained patch embedding')

    # === Output ===
    parser.add_argument('--save_dir', type=str, default=str(ROOT / 'experiments' / 'dualbranch'),
                        help='Directory to save models and logs')
    parser.add_argument('--log_dir', type=str, default=str(ROOT / 'logs' / 'dualbranch'),
                        help='TensorBoard log directory')
    parser.add_argument('--save_model_interval', type=int, default=5000,
                        help='Checkpoint saving frequency')

    # === OT Configuration ===
    parser.add_argument('--ot_method', type=str, default='SW',
                        choices=['none', 'SW', 'MaxSW', 'Sinkhorn'],
                        help='OT method for color loss')
    parser.add_argument('--ot_n_projections', type=int, default=100,
                        help='Number of projections for SW/MaxSW')
    parser.add_argument('--ot_max_points', type=int, default=1024,
                        help='Max spatial points for OT')
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help='Sinkhorn regularization')

    # === Loss Weights ===
    parser.add_argument('--color_weight', type=float, default=10.0,
                        help='Weight for color branch OT loss')
    parser.add_argument('--structure_weight', type=float, default=7.0,
                        help='Weight for structure branch content loss')
    parser.add_argument('--fusion_weight', type=float, default=5.0,
                        help='Weight for fusion perceptual loss')
    parser.add_argument('--identity_weight', type=float, default=50.0,
                        help='Weight for identity regularization')

    # === Training ===
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    parser.add_argument('--max_iter', type=int, default=160000,
                        help='Maximum training iterations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (dual-branch needs more GPU memory)')
    parser.add_argument('--style_weight', type=float, default=3.0,
                        help='Global style weight multiplier')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Global content weight multiplier')
    parser.add_argument('--n_threads', type=int, default=16,
                        help='Number of data loading threads')

    # === Reproducibility ===
    parser.add_argument('--seed', type=int, default=20260907,
                        help='Random seed for reproducible initialization and sampling')

    # === Resume ===
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    return args


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (same as original TransColor)"""
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


def adjust_learning_rate(optimizer, iteration_count, args):
    """Imitating the original paper's learning rate decay"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class TrainDataset(Dataset):
    """Dataset yielding (content, style) image pairs.

    Content images come from content_dir, style images from style_dir.
    Styles wrap around with modulo so a single reference image can serve all content.
    """
    def __init__(self, content_dir, style_dir, transform):
        self.content_paths = sorted(
            list(Path(content_dir).glob('*.png')) + list(Path(content_dir).glob('*.jpg'))
        )
        self.style_paths = sorted(
            list(Path(style_dir).glob('*.png')) + list(Path(style_dir).glob('*.jpg'))
        )
        self.transform = transform

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, index):
        content = Image.open(self.content_paths[index]).convert('RGB')
        style = Image.open(self.style_paths[index % len(self.style_paths)]).convert('RGB')
        return self.transform(content), self.transform(style)


def main():
    args = get_args()

    # Reproducibility: seed every random source before any module or sampler is built.
    set_seed(args.seed)

    # === Device ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DualBranch] Training on {device}")

    # === Directories ===
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    weights = {
        'vgg': Path(args.vgg),
        'decoder': Path(args.decoder),
        'transformer': Path(args.Trans),
        'embedding': Path(args.embedding),
    }
    metadata = build_run_metadata(
        args=args,
        device=device,
        weights=weights,
        content_dir=Path(args.content_dir),
        style_dir=Path(args.style_dir),
    )
    write_run_metadata(save_dir / 'run_metadata.json', metadata)

    # === Dataset ===
    # Same preprocessing as the single-branch trainer for a fair comparison.
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = TrainDataset(args.content_dir, args.style_dir, transform)
    train_iter = iter(DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(train_dataset),
        num_workers=args.n_threads
    ))

    print(f"[DualBranch] Dataset loaded: {len(train_dataset)} pairs")

    # === Model ===
    # Load pre-trained modules
    embedding = PatchEmbed()
    Trans = Transformer()

    decoder.eval()
    vgg.eval()

    # Load the same 160K pretrained modules with an explicit device map.
    decoder.load_state_dict(torch.load(weights['decoder'], map_location=device))
    embedding.load_state_dict(torch.load(weights['embedding'], map_location=device))
    Trans.load_state_dict(torch.load(weights['transformer'], map_location=device))
    vgg.load_state_dict(torch.load(weights['vgg'], map_location=device))

    # Freeze VGG (used only for loss)
    for param in vgg.parameters():
        param.requires_grad = False

    # Create dual-branch network
    network = DualBranchColorTrans(vgg, decoder, embedding, Trans, args).to(device)

    # Configure OT loss
    if args.ot_method != 'none':
        network.set_ot_style_loss(
            method=args.ot_method,
            n_projections=args.ot_n_projections,
            max_points=args.ot_max_points,
            normalize='instance',
            reg=args.ot_reg
        )

    # === Optimizer ===
    optimizer = torch.optim.Adam([
        {'params': network.color_branch.parameters()},
        {'params': network.structure_branch.transformer.parameters()},
        {'params': network.structure_branch.decoder.parameters()},
        {'params': network.fusion_module.parameters()},
    ], lr=args.lr)

    # === Resume ===
    start_iter = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']
        print(f"[DualBranch] Resumed from iteration {start_iter}")

    # === Training Loop ===
    network.train()
    print(f"[DualBranch] Training from iteration {start_iter} to {args.max_iter}")
    print(f"  - Color weight: {args.color_weight}")
    print(f"  - Structure weight: {args.structure_weight}")
    print(f"  - Fusion weight: {args.fusion_weight}")
    print(f"  - Identity weight: {args.identity_weight}")
    print(f"  - OT method: {args.ot_method}")

    for i in range(start_iter, args.max_iter):
        adjust_learning_rate(optimizer, i, args)

        # Get batch
        content_images, style_images = next(train_iter)
        content_images = content_images.to(device)
        style_images = style_images.to(device)

        # Convert to NestedTensor
        samples_c = nested_tensor_from_tensor_list(content_images)
        samples_s = nested_tensor_from_tensor_list(style_images)

        # Forward pass
        (fused, loss_color_ot, loss_structure, loss_fusion,
         loss_identity, color_output, struct_output) = network(
            samples_c, samples_s,
            ot_method=args.ot_method,
            ot_n_proj=args.ot_n_projections
        )

        assert_finite(
            f"dual-branch iteration {i}",
            loss_color_ot,
            loss_structure,
            loss_fusion,
            loss_identity,
        )

        # Total loss
        loss = (
            args.color_weight * loss_color_ot +
            args.structure_weight * loss_structure +
            args.fusion_weight * loss_fusion +
            args.identity_weight * loss_identity
        )
        assert_finite(f"dual-branch total iteration {i}", loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Logging ===
        if i % 50 == 0:
            # Console
            print(f"[Iter {i}/{args.max_iter}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Color: {loss_color_ot.item():.4f} | "
                  f"Struct: {loss_structure.item():.4f} | "
                  f"Fusion: {loss_fusion.item():.4f} | "
                  f"Identity: {loss_identity.item():.4f}")

            # TensorBoard
            writer.add_scalar('loss/total', loss.item(), i)
            writer.add_scalar('loss/color_ot', loss_color_ot.item(), i)
            writer.add_scalar('loss/structure', loss_structure.item(), i)
            writer.add_scalar('loss/fusion', loss_fusion.item(), i)
            writer.add_scalar('loss/identity', loss_identity.item(), i)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)

        # === Save images ===
        if i % 500 == 0:
            output_dir = save_dir / 'samples'
            output_dir.mkdir(exist_ok=True)

            with torch.no_grad():
                # Save: content | style | color_branch | struct_branch | fused
                comparison = torch.cat([
                    content_images[:1],
                    style_images[:1],
                    color_output[:1],
                    struct_output[:1],
                    fused[:1]
                ], dim=0)
                save_image(comparison, output_dir / f'iter_{i:06d}.jpg', nrow=5, normalize=True)

        # === Save checkpoint ===
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state = {
                'iteration': i + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }
            checkpoint_path = save_dir / f'dualbranch_iter_{i+1:06d}.pth'
            torch.save(state, checkpoint_path)
            print(f"[DualBranch] Checkpoint saved: {checkpoint_path}")

    writer.close()
    print("[DualBranch] Training complete!")


if __name__ == '__main__':
    main()
