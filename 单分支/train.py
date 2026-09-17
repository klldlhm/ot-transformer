from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # single/ 与 dual/ 的上一层 = 项目根
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import models.transformer as transformer
import models.TransColor as TransC
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from fair_experiment import (
    assert_finite,
    build_run_metadata,
    set_seed,
    write_run_metadata,
)


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default=str(ROOT / 'datasets' / 'content'), type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default=str(ROOT / 'datasets' / 'style'), type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default=str(ROOT / 'experiments' / 'vgg_normalised.pth'),
                    help='Pretrained VGG checkpoint')
parser.add_argument('--decoder', type=str, default=str(ROOT / 'experiments' / 'decoder_iter_160000.pth'),
                    help='Pretrained decoder checkpoint')
parser.add_argument('--Trans', type=str, default=str(ROOT / 'experiments' / 'transformer_iter_160000.pth'),
                    help='Pretrained Transformer checkpoint')
parser.add_argument('--embedding', type=str, default=str(ROOT / 'experiments' / 'embedding_iter_160000.pth'),
                    help='Pretrained patch-embedding checkpoint')
parser.add_argument('--seed', type=int, default=20260907,
                    help='Random seed for reproducible initialization and sampling')

# Training options
parser.add_argument('--save_dir', default=str(ROOT / 'experiments'),
                    help='Directory to save the model')
parser.add_argument('--log_dir', default=str(ROOT / 'logs'),
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=9000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")

# ---- OT Style Loss Options ----
parser.add_argument('--ot_method', type=str, default='none',
                    choices=['none', 'SW', 'MaxSW', 'Sinkhorn'],
                    help='OT method for style loss (none = AdaIN only)')
parser.add_argument('--ot_n_projections', type=int, default=100,
                    help='Number of random projections for SW/MaxSW')
parser.add_argument('--ot_max_points', type=int, default=1024,
                    help='Max spatial points for OT computation')
parser.add_argument('--ot_weight', type=float, default=1.0,
                    help='Weight for OT loss in style loss mix (0=AdaIN only, 1=OT only)')

args = parser.parse_args()

# Reproducibility: seed every random source before any module or sampler is built.
set_seed(args.seed)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

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
write_run_metadata(Path(args.save_dir) / 'run_metadata.json', metadata)

# Build model from the same 160K pretrained modules
vgg = TransC.vgg
vgg.load_state_dict(torch.load(weights['vgg'], map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = TransC.decoder
decoder.load_state_dict(torch.load(weights['decoder'], map_location=device))
embedding = TransC.PatchEmbed()
embedding.load_state_dict(torch.load(weights['embedding'], map_location=device))

Trans = transformer.Transformer()
Trans.load_state_dict(torch.load(weights['transformer'], map_location=device))
with torch.no_grad():
    network = TransC.ColorTrans(vgg, decoder, embedding, Trans, args)
network.train()

# Configure OT style loss if requested
if args.ot_method != 'none':
    network.set_ot_style_loss(
        method=args.ot_method,
        n_projections=args.ot_n_projections,
        max_points=args.ot_max_points
    )

network.to(device)
network = nn.DataParallel(network, device_ids=[0])

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([
    {'params': network.module.transformer.parameters()},
    {'params': network.module.decode.parameters()},
    {'params': network.module.embedding.parameters()},
], lr=args.lr)

if not os.path.exists(args.save_dir + "/test"):
    os.makedirs(args.save_dir + "/test")

# Log OT configuration
print(f"[OT-TransColor] OT method: {args.ot_method}")
print(f"[OT-TransColor] OT weight: {args.ot_weight}")
print(f"[OT-TransColor] OT n_projections: {args.ot_n_projections}")
print(f"[OT-TransColor] Total loss = Lc*10 + Ls*(1-w)AdaIN + Ls*w*OT + Lid1*100 + Lid2*10")

for i in tqdm(range(args.max_iter)):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    if args.ot_method != 'none':
        out, loss_c, loss_s_adain, loss_s_ot, l_identity1, l_identity2 = network(
            content_images, style_images,
            ot_method=args.ot_method,
            ot_n_proj=args.ot_n_projections
        )
    else:
        # AdaIN-only mode: OT loss is zero, pass 'none' to skip OT computation
        out, loss_c, loss_s_adain, loss_s_ot, l_identity1, l_identity2 = network(
            content_images, style_images,
            ot_method='none'
        )

    assert_finite(
        f"single-branch iteration {i}",
        loss_c,
        loss_s_adain,
        loss_s_ot,
        l_identity1,
        l_identity2,
    )

    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
            args.save_dir, str(i), ".jpg"
        )
        save_image(out, output_name)

    # Loss combination with OT weight
    loss_c = args.content_weight * loss_c

    # Style loss: weighted mix of AdaIN and OT
    if args.ot_method != 'none':
        loss_s = args.style_weight * (
            (1.0 - args.ot_weight) * loss_s_adain + args.ot_weight * loss_s_ot
        )
    else:
        loss_s = args.style_weight * loss_s_adain

    loss = (loss_c * 10) + (loss_s * 1) + (l_identity1 * 100) + (l_identity2 * 10)
    assert_finite(f"single-branch total iteration {i}", loss)

    # Logging
    log_parts = [
        f"T={loss.sum().cpu().detach().numpy():.4f}",
        f"C={loss_c.sum().cpu().detach().numpy():.4f}",
        f"S={loss_s.sum().cpu().detach().numpy():.4f}",
    ]
    if args.ot_method != 'none':
        log_parts.extend([
            f"S_AdaIN={loss_s_adain.sum().cpu().detach().numpy():.4f}",
            f"S_OT={loss_s_ot.sum().cpu().detach().numpy():.4f}",
        ])
    log_parts.extend([
        f"ID1={l_identity1.sum().cpu().detach().numpy():.4f}",
        f"ID2={l_identity2.sum().cpu().detach().numpy():.4f}",
    ])
    print(" | ".join(log_parts))

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_style_adain', loss_s_adain.sum().item(), i + 1)
    if args.ot_method != 'none':
        writer.add_scalar('loss_style_ot', loss_s_ot.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir, i + 1))

        state_dict = network.module.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir, i + 1))

        state_dict = network.module.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir, i + 1))

writer.close()
