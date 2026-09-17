#!/usr/bin/env python3
"""
compare_results.py — Quantitative comparison of dual-branch experiments

Computes metrics for ablation study:
- FID (Fréchet Inception Distance): distribution similarity
- SSIM (Structural Similarity): structure preservation
- LPIPS (Learned Perceptual Image Patch Similarity): perceptual quality
- Color histogram distance: color distribution matching

Usage:
    python compare_results.py \
        --experiments experiments/ablation \
        --reference_dir datasets/PH2 \
        --content_dir datasets/test_content \
        --output results/comparison.csv
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # 项目根（dual/ 的上一层）
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
import csv
from tqdm import tqdm

try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("[WARNING] pytorch-fid not installed. FID computation disabled.")
    print("Install with: pip install pytorch-fid")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARNING] lpips not installed. LPIPS computation disabled.")
    print("Install with: pip install lpips")

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2


def load_image(path: Path, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load image as numpy array"""
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BICUBIC)
    return np.array(img)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images"""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    return psnr(img1, img2, data_range=255)


def compute_color_histogram_distance(img1: np.ndarray, img2: np.ndarray,
                                     bins: int = 64) -> float:
    """
    Compute color histogram distance (Earth Mover's Distance)

    Lower values = more similar color distributions
    """
    # Convert to LAB color space for perceptual uniformity
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

    distance = 0.0
    for channel in range(3):
        hist1 = cv2.calcHist([img1_lab], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([img2_lab], [channel], None, [bins], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # EMD approximation (Wasserstein-1 distance)
        distance += np.sum(np.abs(np.cumsum(hist1) - np.cumsum(hist2)))

    return distance / 3.0


class LPIPSMetric:
    """Wrapper for LPIPS metric"""
    def __init__(self, device='cuda'):
        if not HAS_LPIPS:
            raise RuntimeError("lpips not installed")
        self.model = lpips.LPIPS(net='alex').to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def compute(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute LPIPS distance"""
        img1_t = self.transform(Image.fromarray(img1)).unsqueeze(0).to(self.device)
        img2_t = self.transform(Image.fromarray(img2)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distance = self.model(img1_t, img2_t)

        return distance.item()


def collect_experiment_images(exp_dir: Path, pattern: str = '*_fused.jpg') -> List[Path]:
    """Collect output images from experiment directory"""
    samples_dir = exp_dir / 'samples'
    if not samples_dir.exists():
        # Try test output directory
        samples_dir = exp_dir

    images = sorted(samples_dir.glob(pattern))
    return images


def compute_metrics_for_experiment(
    exp_name: str,
    exp_dir: Path,
    reference_dir: Path,
    content_dir: Path,
    lpips_model: LPIPSMetric = None
) -> dict:
    """Compute all metrics for a single experiment"""

    print(f"\n[{exp_name}] Computing metrics...")

    # Collect images
    output_images = collect_experiment_images(exp_dir)
    if not output_images:
        print(f"  [WARNING] No output images found in {exp_dir}")
        return None

    # Get reference and content images
    reference_images = sorted(Path(reference_dir).glob('*.jpg')) + \
                      sorted(Path(reference_dir).glob('*.png'))
    content_images = sorted(Path(content_dir).glob('*.jpg')) + \
                    sorted(Path(content_dir).glob('*.png'))

    if not reference_images or not content_images:
        print(f"  [WARNING] Missing reference or content images")
        return None

    # Compute metrics
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    color_distances = []

    for i, output_path in enumerate(tqdm(output_images[:min(len(output_images), 100)],
                                        desc=f"  {exp_name}")):
        try:
            # Load images
            output_img = load_image(output_path)

            # Compare with content (for structure preservation)
            if i < len(content_images):
                content_img = load_image(content_images[i])
                ssim_scores.append(compute_ssim(output_img, content_img))
                psnr_scores.append(compute_psnr(output_img, content_img))

                if lpips_model:
                    lpips_scores.append(lpips_model.compute(output_img, content_img))

            # Compare with style (for color distribution)
            style_idx = i % len(reference_images)
            style_img = load_image(reference_images[style_idx])
            color_distances.append(compute_color_histogram_distance(output_img, style_img))

        except Exception as e:
            print(f"  [ERROR] Failed to process {output_path}: {e}")
            continue

    # Aggregate results
    results = {
        'experiment': exp_name,
        'n_images': len(output_images),
        'ssim_mean': np.mean(ssim_scores) if ssim_scores else 0.0,
        'ssim_std': np.std(ssim_scores) if ssim_scores else 0.0,
        'psnr_mean': np.mean(psnr_scores) if psnr_scores else 0.0,
        'psnr_std': np.std(psnr_scores) if psnr_scores else 0.0,
        'color_dist_mean': np.mean(color_distances) if color_distances else 0.0,
        'color_dist_std': np.std(color_distances) if color_distances else 0.0,
    }

    if lpips_scores:
        results['lpips_mean'] = np.mean(lpips_scores)
        results['lpips_std'] = np.std(lpips_scores)

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare dual-branch experiment results')

    parser.add_argument('--experiments', type=str, required=True,
                        help='Base directory containing all experiment outputs')
    parser.add_argument('--reference_dir', type=str, required=True,
                        help='Directory with reference style images')
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory with content test images')
    parser.add_argument('--output', type=str, default=str(ROOT / 'results' / 'comparison.csv'),
                        help='Output CSV file for results')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Compute LPIPS (requires GPU, slow)')

    args = parser.parse_args()

    # Setup
    exp_base = Path(args.experiments)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize LPIPS if requested
    lpips_model = None
    if args.use_lpips and HAS_LPIPS:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Initializing LPIPS on {device}")
        lpips_model = LPIPSMetric(device=device)

    # Find all experiment directories
    exp_dirs = [d for d in exp_base.iterdir() if d.is_dir() and d.name != 'logs']
    exp_dirs = sorted(exp_dirs, key=lambda x: x.name)

    print(f"\n[INFO] Found {len(exp_dirs)} experiments")
    print(f"[INFO] Reference images: {args.reference_dir}")
    print(f"[INFO] Content images: {args.content_dir}")

    # Compute metrics for each experiment
    all_results = []
    for exp_dir in exp_dirs:
        results = compute_metrics_for_experiment(
            exp_dir.name,
            exp_dir,
            args.reference_dir,
            args.content_dir,
            lpips_model
        )

        if results:
            all_results.append(results)

    # Save results to CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n[SUCCESS] Results saved to {output_path}")

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Experiment':<40} {'SSIM':<12} {'PSNR':<12} {'Color Dist':<12}")
        print("-"*80)

        for result in all_results:
            print(f"{result['experiment']:<40} "
                  f"{result['ssim_mean']:.4f}±{result['ssim_std']:.4f}  "
                  f"{result['psnr_mean']:.2f}±{result['psnr_std']:.2f}  "
                  f"{result['color_dist_mean']:.4f}±{result['color_dist_std']:.4f}")

        print("="*80)

        # Highlight best results
        print("\nBEST RESULTS:")
        best_ssim = max(all_results, key=lambda x: x['ssim_mean'])
        best_psnr = max(all_results, key=lambda x: x['psnr_mean'])
        best_color = min(all_results, key=lambda x: x['color_dist_mean'])

        print(f"  Best structure preservation (SSIM): {best_ssim['experiment']} ({best_ssim['ssim_mean']:.4f})")
        print(f"  Best image quality (PSNR): {best_psnr['experiment']} ({best_psnr['psnr_mean']:.2f})")
        print(f"  Best color matching: {best_color['experiment']} ({best_color['color_dist_mean']:.4f})")
    else:
        print("\n[ERROR] No valid results computed")


if __name__ == '__main__':
    main()
