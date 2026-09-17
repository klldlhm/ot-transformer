"""
run_dualbranch_experiments.py — Automated experiment runner for dual-branch ablation studies

Runs systematic comparisons:
1. Baseline single-branch OT-TransColor
2. Dual-branch with different OT methods (SW, MaxSW, Sinkhorn)
3. Ablation: color-only, structure-only, fusion variations
4. Loss weight sensitivity analysis

Usage:
    python run_dualbranch_experiments.py \
        --content_dir ./datasets/train2014 \
        --style_dir ./datasets/PH2 \
        --base_output ./experiments/ablation
"""
import argparse
import subprocess
import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent                 # 项目根（dual/ 的上一层）
sys.path.insert(0, str(_HERE))       # 使 import models / import ot_loss 生效
from dataclasses import dataclass
from typing import List
import json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    name: str
    description: str
    script: str  # 'train.py' or 'train_dualbranch.py'
    args: dict
    max_iter: int = 160000


def get_base_args():
    """Common arguments for all experiments"""
    return {
        'lr': 5e-4,
        'lr_decay': 5e-5,
        'batch_size': 8,
        'n_threads': 16,
        'save_model_interval': 10000,
    }


def define_experiments():
    """Define all ablation experiments"""
    experiments = []

    # ===== Baseline: Single-branch OT-TransColor =====
    experiments.append(ExperimentConfig(
        name='baseline_single_ot_sw',
        description='Single-branch TransColor with SW-OT (baseline)',
        script='train.py',
        args={
            **get_base_args(),
            'ot_method': 'SW',
            'ot_n_projections': 100,
            'ot_weight': 1.0,
            'style_weight': 10.0,
            'content_weight': 7.0,
        }
    ))

    # ===== Dual-branch: Different OT methods =====
    for ot_method in ['SW', 'MaxSW', 'Sinkhorn']:
        experiments.append(ExperimentConfig(
            name=f'dualbranch_ot_{ot_method.lower()}',
            description=f'Dual-branch with {ot_method} for color',
            script='train_dualbranch.py',
            args={
                **get_base_args(),
                'ot_method': ot_method,
                'ot_n_projections': 100 if ot_method != 'Sinkhorn' else None,
                'ot_reg': 0.1 if ot_method == 'Sinkhorn' else None,
                'color_weight': 10.0,
                'structure_weight': 7.0,
                'fusion_weight': 5.0,
                'identity_weight': 50.0,
            }
        ))

    # ===== Ablation: Branch isolation =====
    # Color-only (structure weight = 0)
    experiments.append(ExperimentConfig(
        name='ablation_color_only',
        description='Color branch only (structure and fusion disabled)',
        script='train_dualbranch.py',
        args={
            **get_base_args(),
            'ot_method': 'SW',
            'ot_n_projections': 100,
            'color_weight': 10.0,
            'structure_weight': 0.0,
            'fusion_weight': 0.0,
            'identity_weight': 50.0,
        }
    ))

    # Structure-only (color weight = 0)
    experiments.append(ExperimentConfig(
        name='ablation_structure_only',
        description='Structure branch only (color and fusion disabled)',
        script='train_dualbranch.py',
        args={
            **get_base_args(),
            'ot_method': 'none',
            'color_weight': 0.0,
            'structure_weight': 7.0,
            'fusion_weight': 0.0,
            'identity_weight': 50.0,
        }
    ))

    # Without OT (CNN + AdaIN for color)
    experiments.append(ExperimentConfig(
        name='ablation_no_ot',
        description='Dual-branch without OT (CNN uses AdaIN-style loss)',
        script='train_dualbranch.py',
        args={
            **get_base_args(),
            'ot_method': 'none',
            'color_weight': 10.0,
            'structure_weight': 7.0,
            'fusion_weight': 5.0,
            'identity_weight': 50.0,
        }
    ))

    # ===== Loss weight sensitivity =====
    # Heavy color emphasis
    experiments.append(ExperimentConfig(
        name='weights_heavy_color',
        description='High color weight for vivid colorization',
        script='train_dualbranch.py',
        args={
            **get_base_args(),
            'ot_method': 'SW',
            'ot_n_projections': 100,
            'color_weight': 20.0,  # 2x
            'structure_weight': 5.0,
            'fusion_weight': 5.0,
            'identity_weight': 50.0,
        }
    ))

    # Heavy structure preservation
    experiments.append(ExperimentConfig(
        name='weights_heavy_structure',
        description='High structure weight for faithful anatomy',
        script='train_dualbranch.py',
        args={
            **get_base_args(),
            'ot_method': 'SW',
            'ot_n_projections': 100,
            'color_weight': 10.0,
            'structure_weight': 15.0,  # 2x
            'fusion_weight': 5.0,
            'identity_weight': 50.0,
        }
    ))

    # ===== SW projection count sensitivity =====
    for n_proj in [50, 200]:
        experiments.append(ExperimentConfig(
            name=f'sw_proj_{n_proj}',
            description=f'SW with {n_proj} projections',
            script='train_dualbranch.py',
            args={
                **get_base_args(),
                'ot_method': 'SW',
                'ot_n_projections': n_proj,
                'color_weight': 10.0,
                'structure_weight': 7.0,
                'fusion_weight': 5.0,
                'identity_weight': 50.0,
            },
            max_iter=80000  # Shorter for sensitivity analysis
        ))

    return experiments


def build_command(exp: ExperimentConfig, content_dir: str, style_dir: str,
                  base_output: str) -> List[str]:
    """Build command line for experiment"""
    output_dir = Path(base_output) / exp.name
    log_dir = Path(base_output) / 'logs' / exp.name

    cmd = [
        'python', exp.script,
        '--content_dir', content_dir,
        '--style_dir', style_dir,
        '--save_dir', str(output_dir),
        '--log_dir', str(log_dir),
        '--max_iter', str(exp.max_iter),
    ]

    # Add experiment-specific arguments
    for key, value in exp.args.items():
        if value is None:
            continue
        cmd.append(f'--{key}')
        cmd.append(str(value))

    return cmd


def run_experiment(exp: ExperimentConfig, content_dir: str, style_dir: str,
                   base_output: str, dry_run: bool = False):
    """Execute a single experiment"""
    cmd = build_command(exp, content_dir, style_dir, base_output)

    print(f"\n{'='*80}")
    print(f"Experiment: {exp.name}")
    print(f"Description: {exp.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Skipping actual execution\n")
        return

    # Run experiment
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {exp.name} completed\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {exp.name} failed with code {e.returncode}\n")
        raise


def save_experiment_manifest(experiments: List[ExperimentConfig], output_path: str):
    """Save experiment configurations to JSON for reference"""
    manifest = []
    for exp in experiments:
        manifest.append({
            'name': exp.name,
            'description': exp.description,
            'script': exp.script,
            'args': exp.args,
            'max_iter': exp.max_iter,
        })

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Experiment manifest saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run systematic ablation experiments for dual-branch OT-TransColor'
    )

    parser.add_argument('--content_dir', type=str, required=True,
                        help='Path to content images')
    parser.add_argument('--style_dir', type=str, required=True,
                        help='Path to style images')
    parser.add_argument('--base_output', type=str, default=str(ROOT / 'experiments' / 'ablation'),
                        help='Base output directory for all experiments')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to run (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments and exit')

    args = parser.parse_args()

    # Define experiments
    experiments = define_experiments()

    # List mode
    if args.list:
        print("\nAvailable experiments:\n")
        for i, exp in enumerate(experiments, 1):
            print(f"{i:2d}. {exp.name}")
            print(f"    {exp.description}")
            print(f"    Script: {exp.script}, Iterations: {exp.max_iter}")
            print()
        return

    # Filter experiments if specified
    if args.experiments:
        selected = [exp for exp in experiments if exp.name in args.experiments]
        if not selected:
            print(f"[ERROR] No matching experiments found: {args.experiments}")
            print("Use --list to see available experiments")
            return
        experiments = selected

    # Create output directory
    base_output = Path(args.base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest_path = base_output / 'experiment_manifest.json'
    save_experiment_manifest(experiments, manifest_path)

    # Run experiments
    print(f"\n{'='*80}")
    print(f"Running {len(experiments)} experiments")
    print(f"Base output: {base_output}")
    print(f"{'='*80}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {exp.name}")
        try:
            run_experiment(exp, args.content_dir, args.style_dir,
                          args.base_output, dry_run=args.dry_run)
        except Exception as e:
            print(f"[ERROR] Experiment {exp.name} failed: {e}")
            continue

    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved to: {base_output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
