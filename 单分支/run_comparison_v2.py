"""
run_comparison_v2.py — Rebalanced style loss + OT magnitude normalization.

Fixes two issues in run_comparison.py that caused the output to lose color:

  1. style loss weight was 1 (v1), but the original TransColor paper uses
     style_weight=10. At weight 1 the style signal was ~1000x weaker than the
     identity loss, so the optimizer washed the output toward gray.

  2. OT loss raw magnitude (~0.07-0.27) is ~10-30x smaller than AdaIN (~2-8),
     so even after fix 1 OT would remain underweighted. We normalize OT by a
     FROZEN scalar K = adaIN_0 / ot_0 computed at iter 0, so both configs start
     with the SAME effective style magnitude (~= adaIN_0).

Everything else (content/identity weights, data, training loop) is identical to
run_comparison.py, so the comparison isolates the style-loss variable only.

Output goes to experiments/{name}_v2/ so v1 results stay intact.
"""
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# single/ 的上一层 = 项目根；权重与数据都在根下，故所有路径以 ROOT 为基准。
ROOT = Path(__file__).resolve().parent.parent

import time
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.transformer as transformer_mod
import models.TransColor as TransC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class DummyArgs:
    hidden_dim = 512


def build_model():
    vgg = TransC.vgg
    vgg.load_state_dict(torch.load(str(ROOT / "experiments" / "vgg_normalised.pth"), map_location=device))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    decoder = TransC.decoder
    embedding = TransC.PatchEmbed()
    Trans = transformer_mod.Transformer()
    net = TransC.ColorTrans(vgg, decoder, embedding, Trans, DummyArgs())
    for attr_name, path in [
        ("decode",      str(ROOT / "experiments" / "decoder_iter_160000.pth")),
        ("transformer", str(ROOT / "experiments" / "transformer_iter_160000.pth")),
        ("embedding",   str(ROOT / "experiments" / "embedding_iter_160000.pth")),
    ]:
        sd = torch.load(path, map_location=device)
        getattr(net, attr_name).load_state_dict(sd)
    net.to(device)
    return net


# ---- data (identical to v1) ----
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

content_imgs = []
for f in sorted(os.listdir(ROOT / "datasets" / "content")):
    if f.endswith(('.png', '.jpg', '.jpeg')):
        content_imgs.append(tf(Image.open(ROOT / "datasets" / "content" / f).convert("RGB")))
style_img = tf(Image.open(ROOT / "datasets" / "style" / "head.jpg").convert("RGB"))
print(f"Loaded {len(content_imgs)} content images + 1 style image")

# ---- training config ----
MAX_ITER = 500
SAVE_EVERY = 100
LR = 5e-4
STYLE_WEIGHT = 10.0   # v1 used 1.0; original paper uses style_weight=10

experiments = [
    {"name": "baseline_adain", "ot_method": "none", "ot_n_proj": 50},
    {"name": "ot_sw_50",      "ot_method": "SW",    "ot_n_proj": 50},
]

results = {}

for exp in experiments:
    exp_name = exp["name"]
    ot_method = exp["ot_method"]
    out_dir = ROOT / "experiments" / f"{exp_name}_v2"
    os.makedirs(out_dir / "test", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running: {exp_name} (v2)")
    print(f"{'='*60}")

    model = build_model()
    model.train()
    if ot_method != "none":
        model.set_ot_style_loss(method=ot_method, n_projections=exp["ot_n_proj"])

    optimizer = torch.optim.Adam([
        {'params': model.transformer.parameters()},
        {'params': model.decode.parameters()},
        {'params': model.embedding.parameters()},
    ], lr=LR)

    loss_history = []
    t_start = time.time()
    ot_scale = 1.0   # frozen magnitude-normalization scalar for OT (computed at iter 0)

    for i in range(MAX_ITER):
        c = content_imgs[i % len(content_imgs)].unsqueeze(0).to(device)
        s = style_img.unsqueeze(0).to(device)

        out, lc, ls_adain, ls_ot, lid1, lid2 = model(
            [c.squeeze(0)], [s.squeeze(0)],
            ot_method=ot_method, ot_n_proj=exp["ot_n_proj"]
        )

        if ot_method != "none":
            if i == 0:
                ot_scale = ls_adain.sum().item() / (ls_ot.sum().item() + 1e-8)
                print(f"  [OT] magnitude normalization K = {ot_scale:.2f} "
                      f"(adaIN_0={ls_adain.sum().item():.3f} / ot_0={ls_ot.sum().item():.4f})")
            loss_s = ls_ot * ot_scale
        else:
            loss_s = ls_adain

        total_loss = lc * 10 + loss_s * STYLE_WEIGHT + lid1 * 100 + lid2 * 10
        total_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_vals = {
            "iter": i,
            "total": total_loss.sum().item(),
            "content": lc.sum().item(),
            "style_adain": ls_adain.sum().item(),
            "style_ot": ls_ot.sum().item(),
            "style_used": loss_s.sum().item(),   # the style term actually optimized
            "ot_scale": ot_scale,
            "identity1": lid1.sum().item(),
            "identity2": lid2.sum().item(),
        }
        loss_history.append(loss_vals)

        if i % SAVE_EVERY == 0 or i == MAX_ITER - 1:
            save_image(out, out_dir / "test" / f"iter_{i:04d}.png")
            print(f"  iter {i:4d} | total={loss_vals['total']:.2f} | "
                  f"C={loss_vals['content']:.4f} | S_used={loss_vals['style_used']:.4f} | "
                  f"S_adain={loss_vals['style_adain']:.3f} | S_ot={loss_vals['style_ot']:.4f} | "
                  f"id1={loss_vals['identity1']:.4f} | id2={loss_vals['identity2']:.3f}")

    elapsed = time.time() - t_start
    save_image(out, out_dir / "test_final.png")
    results[exp_name] = loss_history
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"  {exp_name} done in {elapsed:.0f}s ({elapsed/MAX_ITER:.2f}s/iter)")

    del model
    torch.cuda.empty_cache()

# ---- comparison report ----
print(f"\n{'='*60}")
print("  COMPARISON REPORT (v2)")
print(f"{'='*60}")

for exp_name, hist in results.items():
    first, final = hist[0], hist[-1]
    print(f"\n  {exp_name}:")
    print(f"    Total:       {first['total']:.2f} -> {final['total']:.2f}")
    print(f"    Content:     {first['content']:.4f} -> {final['content']:.4f}")
    print(f"    Style_used:  {first['style_used']:.4f} -> {final['style_used']:.4f}")
    print(f"    Style_adain: {first['style_adain']:.3f} -> {final['style_adain']:.3f}")
    print(f"    Style_ot:    {first['style_ot']:.4f} -> {final['style_ot']:.4f}")

print("\nDone! Output in experiments/{baseline_adain,ot_sw_50}_v2/")
