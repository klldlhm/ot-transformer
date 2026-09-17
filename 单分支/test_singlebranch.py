"""
test_singlebranch.py — Inference for the original single-branch dual-transformer (ColorTrans)

Loads the transformer/decoder/embedding checkpoints produced by train.py and colorizes
the same content images used by the dual-branch, so the two can be compared fairly.

Usage:
    python test_singlebranch.py \
        --content datasets/content \
        --style datasets/style/head.jpg \
        --save_dir experiments/baseline_smoke \
        --output results/singlebranch \
        --ot_method SW
"""
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # single/ 与 dual/ 的上一层 = 项目根
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.transformer import Transformer
from models.TransColor import ColorTrans, decoder
from models.TransColor import vgg as vgg_net
from models.TransColor import PatchEmbed
from util.misc import nested_tensor_from_tensor_list


def get_args():
    parser = argparse.ArgumentParser(description='Single-branch ColorTrans inference')
    parser.add_argument('--content', type=str, required=True,
                        help='Content image or directory')
    parser.add_argument('--style', type=str, required=True,
                        help='Style image or directory')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Folder with transformer/decoder/embedding_iter_N.pth')
    parser.add_argument('--ckpt_iter', type=int, default=3000,
                        help='Checkpoint iteration number (e.g. 3000 or 160000)')
    parser.add_argument('--output', type=str, default=str(ROOT / 'results' / 'singlebranch'))
    parser.add_argument('--vgg', type=str, default=str(ROOT / 'experiments' / 'vgg_normalised.pth'))
    parser.add_argument('--ot_method', type=str, default='SW',
                        choices=['none', 'SW', 'MaxSW', 'Sinkhorn'])
    parser.add_argument('--ot_n_projections', type=int, default=100)
    return parser.parse_args()


def load_image(path, device):
    tf = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    return tf(Image.open(path).convert('RGB')).unsqueeze(0).to(device)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.save_dir)

    # Load pretrained / trained modules (mirrors train.py construction)
    vgg = vgg_net
    vgg.load_state_dict(torch.load(args.vgg, map_location=device))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    for p in vgg.parameters():
        p.requires_grad = (False)
    vgg.eval()

    dec = decoder
    dec.load_state_dict(torch.load(ckpt / f'decoder_iter_{args.ckpt_iter}.pth', map_location=device))
    dec.eval()

    embedding = PatchEmbed()
    embedding.load_state_dict(torch.load(ckpt / f'embedding_iter_{args.ckpt_iter}.pth', map_location=device))

    Trans = Transformer()
    Trans.load_state_dict(torch.load(ckpt / f'transformer_iter_{args.ckpt_iter}.pth', map_location=device))

    # ColorTrans expects an args object; it is unused internally.
    import argparse as _arg
    args_ns = _arg.Namespace()
    net = ColorTrans(vgg, dec, embedding, Trans, args_ns).to(device)
    net.eval()
    if args.ot_method != 'none':
        net.set_ot_style_loss(method=args.ot_method,
                              n_projections=args.ot_n_projections, max_points=1024)

    content_path = Path(args.content)
    style_path = Path(args.style)
    content_imgs = (sorted(content_path.glob('*.png')) + sorted(content_path.glob('*.jpg'))
                    if content_path.is_dir() else [content_path])
    style_imgs = (sorted(style_path.glob('*.png')) + sorted(style_path.glob('*.jpg'))
                  if style_path.is_dir() else [style_path])

    print(f"[SingleBranch] {len(content_imgs)} content, {len(style_imgs)} style -> {out_dir}")
    with torch.no_grad():
        for cp in content_imgs:
            for sp in style_imgs:
                c = load_image(cp, device)
                s = load_image(sp, device)
                # ColorTrans.forward -> (Ics, loss_c, loss_s_adain, loss_s_ot, id1, id2)
                out = net(c, s, ot_method=args.ot_method,
                          ot_n_proj=args.ot_n_projections)[0]
                name = f"{cp.stem}_stylized_by_{sp.stem}.jpg"
                save_image(out, out_dir / name, normalize=True)
    print("[SingleBranch] inference complete.")


if __name__ == '__main__':
    main()
