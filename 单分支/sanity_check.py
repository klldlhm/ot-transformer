"""
sanity_check.py — 验证数据加载 + 权重加载 + 推理 + OT损失，然后跑几轮训练。
"""
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# single/ 的上一层 = 项目根；权重与数据都在根下，故所有路径以 ROOT 为基准。
ROOT = Path(__file__).resolve().parent.parent

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.transformer as transformer_mod
import models.TransColor as TransC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- 1. 构建模型 ----
vgg = TransC.vgg
vgg.load_state_dict(torch.load(str(ROOT / "experiments" / "vgg_normalised.pth"), map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])
decoder = TransC.decoder
embedding = TransC.PatchEmbed()
Trans = transformer_mod.Transformer()

class DummyArgs:
    hidden_dim = 512

network = TransC.ColorTrans(vgg, decoder, embedding, Trans, DummyArgs())
network.to(device)

# 加载预训练权重
import collections
for attr_name, path in [
    ("decode",      str(ROOT / "experiments" / "decoder_iter_160000.pth")),
    ("transformer", str(ROOT / "experiments" / "transformer_iter_160000.pth")),
    ("embedding",   str(ROOT / "experiments" / "embedding_iter_160000.pth")),
]:
    sd = torch.load(path, map_location=device)
    new_sd = collections.OrderedDict()
    for k, v in sd.items():
        new_sd[k] = v
    getattr(network, attr_name).load_state_dict(new_sd)
    print(f"  Loaded {attr_name}: {path}")

network.eval()
print("Model loaded OK")

# ---- 2. 加载数据 ----
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

content_files = sorted([f for f in os.listdir(ROOT / "datasets" / "content") if f.endswith(('.png','.jpg','.jpeg'))])
style_files   = sorted([f for f in os.listdir(ROOT / "datasets" / "style")   if f.endswith(('.png','.jpg','.jpeg'))])

print(f"Content images: {len(content_files)}")
print(f"Style images:   {len(style_files)}")

# 取第一张内容图和风格图做推理
content = tf(Image.open(ROOT / "datasets" / "content" / content_files[0]).convert("RGB"))
style   = tf(Image.open(ROOT / "datasets" / "style" / style_files[0]).convert("RGB"))
content = content.unsqueeze(0).to(device)
style   = style.unsqueeze(0).to(device)
print(f"Content shape: {content.shape}, Style shape: {style.shape}")

# ---- 3. 推理测试 (AdaIN) ----
with torch.no_grad():
    out, lc, ls_adain, ls_ot, lid1, lid2 = network(content, style, ot_method='none')
print(f"AdaIN mode: Ics shape={out.shape}, Lc={lc.item():.4f}, Ls={ls_adain.item():.4f}")

os.makedirs(ROOT / "experiments" / "sanity_test", exist_ok=True)
save_image(out, ROOT / "experiments" / "sanity_test" / "test_adain.png")
print(f"Saved: {ROOT / 'experiments' / 'sanity_test' / 'test_adain.png'}")

# ---- 4. 推理测试 (OT-SW) ----
with torch.no_grad():
    out, lc, ls_adain, ls_ot, lid1, lid2 = network(content, style, ot_method='SW', ot_n_proj=50)
print(f"OT-SW mode: Ics shape={out.shape}, Lc={lc.item():.4f}, LsAdaIN={ls_adain.item():.4f}, LsOT={ls_ot.item():.4f}")

save_image(out, ROOT / "experiments" / "sanity_test" / "test_ot_sw.png")
print(f"Saved: {ROOT / 'experiments' / 'sanity_test' / 'test_ot_sw.png'}")

# ---- 5. 快速训练循环 (20 iter) ----
print("\n--- Quick training test (20 iterations) ---")
network.train()

optimizer = torch.optim.Adam([
    {'params': network.transformer.parameters()},
    {'params': network.decode.parameters()},
    {'params': network.embedding.parameters()},
], lr=1e-4)

for i in range(20):
    # random content image + same style image
    idx = i % len(content_files)
    c_img = tf(Image.open(ROOT / "datasets" / "content" / content_files[idx]).convert("RGB"))
    s_img = tf(Image.open(ROOT / "datasets" / "style" / style_files[0]).convert("RGB"))

    c = [c_img.to(device)]  # pass as list for nested_tensor_from_tensor_list
    s = [s_img.to(device)]

    out, lc, ls_adain, ls_ot, lid1, lid2 = network(c, s, ot_method='SW', ot_n_proj=50)

    loss = lc * 10 + ls_ot * 1 + lid1 * 100 + lid2 * 10
    loss.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        print(f"  iter {i:3d}: total={loss.sum().item():.4f}, "
              f"C={lc.sum().item():.4f}, OT={ls_ot.sum().item():.4f}, "
              f"id1={lid1.sum().item():.4f}, id2={lid2.sum().item():.4f}")

print("\nAll checks passed - ready for training!")
