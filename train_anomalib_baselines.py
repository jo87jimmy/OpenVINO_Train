#!/usr/bin/env python
"""
使用 Anomalib 訓練 Baseline 模型 (PatchCore, CFlow, RD4AD, EfficientAD)
在 MVTec Anomaly Detection 資料集上進行簡單訓練，產生權重檔供 benchmark 使用。

Usage:
    python train_anomalib_baselines.py --data_root ./data-mvtec/mvtec --output_dir ./anomalib_results
    python train_anomalib_baselines.py --data_root ./data-mvtec/mvtec --models PatchCore RD4AD
    python train_anomalib_baselines.py --data_root ./data-mvtec/mvtec --categories bottle carpet
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

import torch

# ── Anomalib imports ──
from anomalib.data import MVTec
from anomalib.models import Patchcore, Cflow, ReverseDistillation, EfficientAd
from anomalib.engine import Engine

# =====================================================================
# 常數定義
# =====================================================================
MVTEC_CATEGORIES = [
    "capsule", "bottle", "carpet", "leather", "pill",
    "transistor", "tile", "cable", "zipper", "toothbrush",
    "metal_nut", "hazelnut", "screw", "grid", "wood",
]

# 模型名稱 → (Anomalib 類別, 預設 epoch 數)
# 注意: PatchCore 是 memory-bank 方法，不需要多 epoch 訓練
MODEL_REGISTRY = {
    "PatchCore": (Patchcore, 1),
    "CFlow": (Cflow, 50),
    "RD4AD": (ReverseDistillation, 200),
    "EfficientAD": (EfficientAd, 70),
}

IMG_SIZE = (256, 256)


# =====================================================================
# 訓練函數
# =====================================================================
def train_single(model_name, category, data_root, output_dir, device_args):
    """
    訓練單一模型在單一類別上。
    回傳: best checkpoint 路徑 (str) 或 None（若失敗）
    """
    model_class, max_epochs = MODEL_REGISTRY[model_name]

    print(f"\n{'─' * 60}")
    print(f"  訓練 {model_name} — 類別: {category} — Epochs: {max_epochs}")
    print(f"{'─' * 60}")

    # ── 建立模型 ──
    model = model_class()

    # ── 建立 MVTec DataModule ──
    datamodule = MVTec(
        root=data_root,
        category=category,
        image_size=IMG_SIZE,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    # ── 訓練輸出路徑 ──
    run_dir = os.path.join(output_dir, model_name, category)
    os.makedirs(run_dir, exist_ok=True)

    # ── 建立 Engine (基於 Lightning Trainer) ──
    engine = Engine(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=run_dir,
        **device_args,
    )

    # ── 訓練 ──
    t0 = time.time()
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"  ❌ 訓練失敗: {e}")
        return None
    elapsed = time.time() - t0
    print(f"  ✅ 訓練完成 — 耗時: {elapsed:.1f}s")

    # ── 取得 best checkpoint 路徑 ──
    ckpt_path = engine.trainer.checkpoint_callback.best_model_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        # fallback: 搜尋 checkpoint 目錄
        ckpt_dir = Path(run_dir)
        ckpts = sorted(ckpt_dir.rglob("*.ckpt"), key=os.path.getmtime, reverse=True)
        ckpt_path = str(ckpts[0]) if ckpts else None

    if ckpt_path:
        # 複製到標準路徑方便後續使用
        standard_path = os.path.join(output_dir, "checkpoints", model_name, f"{category}.ckpt")
        os.makedirs(os.path.dirname(standard_path), exist_ok=True)
        shutil.copy2(ckpt_path, standard_path)
        print(f"  📦 Checkpoint 已存至: {standard_path}")
        return standard_path
    else:
        print(f"  ⚠️ 未找到 checkpoint 檔案")
        return None


def train_all(args):
    """訓練所有選定模型在所有選定類別上。"""
    models = args.models if args.models else list(MODEL_REGISTRY.keys())
    categories = args.categories if args.categories else MVTEC_CATEGORIES

    # 驗證模型名稱
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"❌ 未知模型: {m}")
            print(f"   可用模型: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    # 裝置設定
    device_args = {}
    if args.precision:
        device_args["precision"] = args.precision

    print("=" * 70)
    print("  Anomalib Baseline 訓練")
    print("=" * 70)
    print(f"  模型: {models}")
    print(f"  類別: {categories}")
    print(f"  資料集路徑: {args.data_root}")
    print(f"  輸出路徑: {args.output_dir}")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # 記錄所有 checkpoint 路徑
    checkpoint_registry = {}

    total_start = time.time()
    for model_name in models:
        checkpoint_registry[model_name] = {}
        for category in categories:
            ckpt = train_single(
                model_name, category, args.data_root, args.output_dir, device_args
            )
            checkpoint_registry[model_name][category] = ckpt

    total_elapsed = time.time() - total_start

    # ── 儲存 checkpoint registry ──
    registry_path = os.path.join(args.output_dir, "checkpoint_registry.json")
    with open(registry_path, "w") as f:
        json.dump(checkpoint_registry, f, indent=2)
    print(f"\n📋 Checkpoint 路徑已記錄: {registry_path}")

    # ── 總結 ──
    print(f"\n{'=' * 70}")
    print(f"  訓練完成總結")
    print(f"{'=' * 70}")
    print(f"  總耗時: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    success_count = 0
    fail_count = 0
    for model_name in models:
        for category in categories:
            if checkpoint_registry[model_name][category]:
                success_count += 1
            else:
                fail_count += 1
                print(f"  ❌ 失敗: {model_name} / {category}")

    print(f"  成功: {success_count} / {success_count + fail_count}")
    if fail_count > 0:
        print(f"  失敗: {fail_count}")
    print(f"{'=' * 70}")

    return checkpoint_registry


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Anomalib 訓練 Baseline 模型 (PatchCore, CFlow, RD4AD, EfficientAD)"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data-mvtec/mvtec",
        help="MVTec 資料集根目錄 (包含各類別資料夾)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./anomalib_results",
        help="訓練結果與 checkpoint 輸出目錄",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="要訓練的模型 (預設全部): PatchCore CFlow RD4AD EfficientAD",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="要訓練的 MVTec 類別 (預設全部15類)",
    )
    parser.add_argument(
        "--precision", type=str, default=None,
        help="訓練精度 (e.g., 16-mixed, 32)",
    )

    args = parser.parse_args()
    train_all(args)
