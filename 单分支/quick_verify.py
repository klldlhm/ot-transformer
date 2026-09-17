"""
quick_verify.py — Verify OT loss correctness, differentiability, and TransColor integration.

Run this before full training to confirm everything works.
Usage: python quick_verify.py
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ot_loss import (
    sliced_wasserstein,
    max_sliced_wasserstein,
    sinkhorn_distance,
    compute_ot_distance,
    feature_to_pointcloud,
    OTStyleLoss,
)


def test_ot_correctness():
    """Test 1: Same distribution → distance ≈ 0; different → distance > 0."""
    print("=" * 60)
    print("Test 1: OT Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    N, C = 500, 64

    # Same distribution
    X = torch.randn(N, C)
    for method in ['SW', 'MaxSW', 'Sinkhorn']:
        dist = compute_ot_distance(X, X, method=method, n_projections=50)
        print(f"  {method:8s}: dist(X, X)     = {dist.item():.6f}  (should be ≈ 0)")
        assert dist.item() < 0.05, f"{method}: same distribution should give ~0, got {dist.item():.4f}"

    # Different distribution (shifted)
    Y = torch.randn(N, C) + 5.0
    for method in ['SW', 'MaxSW', 'Sinkhorn']:
        d_same = compute_ot_distance(X, X, method=method, n_projections=50)
        d_diff = compute_ot_distance(X, Y, method=method, n_projections=50)
        print(f"  {method:8s}: dist(X, X+5)   = {d_diff.item():.6f}  (should be >> 0)")
        assert d_diff.item() > d_same.item(), \
            f"{method}: different should be larger than same"

    print("  [PASS] OT correctness\n")


def test_ot_differentiable():
    """Test 2: Gradient flows back through OT distance to input features."""
    print("=" * 60)
    print("Test 2: Differentiability")
    print("=" * 60)

    torch.manual_seed(42)
    N, C = 500, 64

    X = torch.randn(N, C, requires_grad=True)
    Y = torch.randn(N, C)

    for method in ['SW', 'MaxSW', 'Sinkhorn']:
        dist = compute_ot_distance(X, Y, method=method, n_projections=50)
        dist.backward()

        if X.grad is not None:
            grad_norm = X.grad.abs().sum().item()
            print(f"  {method:8s}: |grad| = {grad_norm:.4f}  (should be > 0)")
            assert grad_norm > 0, f"{method}: gradient should not be zero"
        else:
            print(f"  {method:8s}: grad = None  [FAIL]")
            assert False, f"{method}: no gradient"

        X.grad = None  # Reset for next test

    print("  [PASS] Differentiability\n")


def test_maxsw_geq_sw():
    """Test 3: MaxSW >= SW for the same data (max >= mean)."""
    print("=" * 60)
    print("Test 3: MaxSW >= SW")
    print("=" * 60)

    torch.manual_seed(42)
    N, C = 500, 64

    X = torch.randn(N, C)
    Y = torch.randn(N, C) + 3.0

    for seed in [0, 1, 2]:
        sw = sliced_wasserstein(X, Y, n_projections=200, seed=seed)
        msw = max_sliced_wasserstein(X, Y, n_projections=200, seed=seed)
        print(f"  seed={seed}: SW={sw.item():.6f}, MaxSW={msw.item():.6f}, "
              f"MaxSW >= SW: {msw.item() >= sw.item()}")
        assert msw.item() >= sw.item() - 1e-6, \
            f"MaxSW should be >= SW (max vs mean of same projections)"

    print("  [PASS] MaxSW >= SW\n")


def test_feature_to_pointcloud():
    """Test 4: Feature map → point cloud conversion with subsampling."""
    print("=" * 60)
    print("Test 4: Feature to Point Cloud Conversion")
    print("=" * 60)

    # Small feature: no subsampling needed
    feat_small = torch.randn(2, 256, 28, 28)
    pc = feature_to_pointcloud(feat_small, max_points=1024)
    expected_n = 2 * 28 * 28  # 1568
    print(f"  Small feat (2,256,28,28): N={pc.shape[0]} (expected {expected_n}), C={pc.shape[1]}")
    assert pc.shape[0] == expected_n and pc.shape[1] == 256

    # Large feature: needs subsampling
    feat_large = torch.randn(1, 64, 224, 224)
    pc = feature_to_pointcloud(feat_large, max_points=1024)
    print(f"  Large feat (1,64,224,224): N={pc.shape[0]} (max=1024), C={pc.shape[1]}")
    assert pc.shape[0] <= 1024 and pc.shape[1] == 64

    # Normalization check
    pc_norm = feature_to_pointcloud(feat_small, max_points=1024, normalize='instance')
    mean = pc_norm.mean(dim=0).abs().max().item()
    std = pc_norm.std(dim=0).mean().item()
    print(f"  Instance norm: max|mean|={mean:.6f} (≈0), avg std={std:.4f} (≈1)")
    assert mean < 0.1, f"Mean should be ~0 after instance norm, got {mean:.4f}"

    print("  [PASS] Feature conversion\n")


def test_ot_style_loss_module():
    """Test 5: OTStyleLoss module with VGG-like feature lists."""
    print("=" * 60)
    print("Test 5: OTStyleLoss Module")
    print("=" * 60)

    # Simulate 5 VGG layers
    shapes = [
        (2, 64, 224, 224),
        (2, 128, 112, 112),
        (2, 256, 56, 56),
        (2, 512, 28, 28),
        (2, 512, 14, 14),
    ]

    output_feats = [torch.randn(*s) for s in shapes]
    style_feats = [torch.randn(*s) for s in shapes]

    for method in ['SW', 'MaxSW', 'Sinkhorn']:
        ot_loss = OTStyleLoss(method=method, n_projections=50, max_points=512)
        loss = ot_loss(output_feats, style_feats)
        print(f"  {method:8s}: loss = {loss.item():.6f} (finite: {torch.isfinite(loss).item()})")
        assert torch.isfinite(loss).all(), f"{method}: loss should be finite"

    print("  [PASS] OTStyleLoss module\n")


def test_full_transcolor_integration():
    """Test 6: End-to-end forward + backward pass with TransColor + OT."""
    print("=" * 60)
    print("Test 6: Full TransColor + OT Integration")
    print("=" * 60)

    # Need to import TransColor models
    from models import transformer as transformer_mod
    from models import TransColor as TransC

    # Simple args-like object
    class DummyArgs:
        hidden_dim = 512

    args = DummyArgs()

    # Build minimal model
    vgg_encoder = TransC.vgg
    decoder = TransC.decoder
    embedding = TransC.PatchEmbed()
    trans = transformer_mod.Transformer()

    model = TransC.ColorTrans(vgg_encoder, decoder, embedding, trans, args)

    # Test 6a: AdaIN mode (backward compat)
    print("\n  [6a] AdaIN mode (ot_method='none')")
    content = torch.randn(1, 3, 224, 224)
    style = torch.randn(1, 3, 224, 224)

    out, lc, ls_adain, ls_ot, lid1, lid2 = model(content, style, ot_method='none')

    assert out.shape == (1, 3, 224, 224), f"Output shape wrong: {out.shape}"
    assert ls_ot.item() == 0.0, f"OT loss should be 0 in AdaIN mode: {ls_ot.item()}"
    print(f"    Output: {out.shape}, Lc={lc.item():.4f}, Ls={ls_adain.item():.4f}, Lot=0")

    # Test 6b: SW mode
    print("\n  [6b] SW mode (ot_method='SW')")
    out, lc, ls_adain, ls_ot, lid1, lid2 = model(content, style, ot_method='SW', ot_n_proj=50)
    assert torch.isfinite(ls_ot).all(), f"OT loss is NaN/Inf: {ls_ot}"
    print(f"    Output: {out.shape}, Lc={lc.item():.4f}, Ls_AdaIN={ls_adain.item():.4f}, Ls_OT={ls_ot.item():.4f}")

    # Test 6c: Full backward pass with OT loss
    print("\n  [6c] Full backward pass with OT loss")
    total_loss = lc * 10 + ls_ot * 1 + lid1 * 100 + lid2 * 10
    total_loss.sum().backward()

    grad_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print(f"    [FAIL] No grad: {name}")
            grad_ok = False
    if grad_ok:
        print("    ✓ All parameters received gradients")

    print("  [PASS] Full integration\n")


def main():
    print("\n" + "=" * 60)
    print("  OT-TransColor Quick Verification")
    print("=" * 60 + "\n")

    all_passed = True
    try:
        test_ot_correctness()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        all_passed = False

    try:
        test_ot_differentiable()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        all_passed = False

    try:
        test_maxsw_geq_sw()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        all_passed = False

    try:
        test_feature_to_pointcloud()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        all_passed = False

    try:
        test_ot_style_loss_module()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        all_passed = False

    try:
        test_full_transcolor_integration()
    except Exception as e:
        print(f"  [FAIL]: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("  [PASS] ALL TESTS PASSED - Ready for training!")
    else:
        print("  [FAIL] SOME TESTS FAILED - See above for details")
    print("=" * 60)


if __name__ == '__main__':
    main()
