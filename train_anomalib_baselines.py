#!/usr/bin/env python
"""
使用 Anomalib 訓練 Baseline 模型 (PatchCore, CFlow, RD4AD, EfficientAD)
在 MVTec Anomaly Detection 資料集上進行簡單訓練，產生權重檔供 benchmark 使用。

支援 Kaggle Notebook 環境自動偵測。

Usage (CLI):
    python train_anomalib_baselines.py --data_root ./mvtec --output_dir ./anomalib_results
    python train_anomalib_baselines.py --data_root ./mvtec --models PatchCore RD4AD
    python train_anomalib_baselines.py --data_root ./mvtec --categories bottle carpet

Usage (Kaggle Notebook):
    from train_anomalib_baselines import train_all_kaggle
    registry = train_all_kaggle(
        models=["PatchCore", "RD4AD"],
        categories=["bottle", "carpet"],
    )
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

# ── Anomalib imports ──
from anomalib.data import MVTecAD
from anomalib.models import Patchcore, Cflow, ReverseDistillation, EfficientAd
from anomalib.engine import Engine

# ── Torchvision transforms ──
from torchvision.transforms.v2 import Resize
import torch

# =====================================================================
# 環境偵測
# =====================================================================
IS_KAGGLE = os.path.exists("/kaggle/working")

# Kaggle 預設路徑
KAGGLE_DATA_ROOT = "/kaggle/input/mvtec-ad"  # 依實際 dataset slug 調整
KAGGLE_OUTPUT_DIR = "/kaggle/working/anomalib_results"

# =====================================================================
# 常數定義
# =====================================================================
MVTEC_CATEGORIES = [
    "capsule",
    "bottle",
    "carpet",
    "leather",
    "pill",
    "transistor",
    "tile",
    "cable",
    "zipper",
    "toothbrush",
    "metal_nut",
    "hazelnut",
    "screw",
    "grid",
    "wood",
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

# 每個模型的建議 batch size（EfficientAD 必須為 1，其餘可較大）
MODEL_BATCH_SIZE = {
    "PatchCore": 32,
    "CFlow": 16,
    "RD4AD": 32,
    "EfficientAD": 1,
}

# Kaggle 環境建議的保守參數 (T4/P100 16GB VRAM, /dev/shm 受限)
KAGGLE_BATCH_SIZE = 16
KAGGLE_NUM_WORKERS = 2


# =====================================================================
# 訓練函數
# =====================================================================
def train_single(
    model_name,
    category,
    data_root,
    output_dir,
    device_args,
    batch_size=32,
    num_workers=4,
):
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

    # ── 建立 MVTecAD DataModule ──
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        augmentations=Resize(IMG_SIZE),
    )

    # ── EfficientAD 強制 batch_size=1（部分 anomalib 版本忽略建構參數）──
    if model_name == "EfficientAD":
        datamodule.train_batch_size = 1

    # ── 若本地資料已存在，跳過下載 (避免 HF Hub 404 錯誤) ──
    local_data_path = Path(data_root) / category
    if local_data_path.exists() and (local_data_path / "train").exists():
        print(f"  📂 使用本地資料: {local_data_path}")
        datamodule.prepare_data = lambda: None

    # ── 訓練輸出路徑 ──
    run_dir = os.path.join(output_dir, model_name, category)
    os.makedirs(run_dir, exist_ok=True)

    # ── 建立 Engine (基於 Lightning Trainer) ──
    # PatchCore 是 memory-bank 方法，無 optimizer，不支援 AMP precision
    engine_device_args = {
        k: v
        for k, v in device_args.items()
        if not (k == "precision" and model_name == "PatchCore")
    }
    engine = Engine(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=run_dir,
        **engine_device_args,
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
        standard_path = os.path.join(
            output_dir, "checkpoints", model_name, f"{category}.ckpt"
        )
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
    print(
        f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    print("=" * 70)

    # 記錄所有 checkpoint 路徑
    checkpoint_registry = {}

    # Kaggle 環境使用保守參數
    num_workers = KAGGLE_NUM_WORKERS if IS_KAGGLE else 4

    total_start = time.time()
    for model_name in models:
        checkpoint_registry[model_name] = {}
        # 根據模型選擇 batch size（EfficientAD 必須為 1）
        default_bs = MODEL_BATCH_SIZE.get(model_name, 32)
        batch_size = min(default_bs, KAGGLE_BATCH_SIZE) if IS_KAGGLE else default_bs
        for category in categories:
            ckpt = train_single(
                model_name,
                category,
                args.data_root,
                args.output_dir,
                device_args,
                batch_size=batch_size,
                num_workers=num_workers,
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
# Kaggle Notebook 入口 (不需要 argparse)
# =====================================================================
def train_all_kaggle(
    data_root=None,
    output_dir=None,
    models=None,
    categories=None,
    precision=None,
):
    """
    Kaggle Notebook 專用入口函數，無需 argparse。

    Usage (在 Kaggle Notebook cell 中):
        from train_anomalib_baselines import train_all_kaggle

        # 訓練指定模型與類別
        registry = train_all_kaggle(
            models=["PatchCore", "RD4AD"],
            categories=["bottle", "carpet"],
        )

        # 訓練全部 (注意時間限制)
        registry = train_all_kaggle()
    """
    args = argparse.Namespace(
        data_root=data_root or (KAGGLE_DATA_ROOT if IS_KAGGLE else "./mvtec"),
        output_dir=output_dir
        or (KAGGLE_OUTPUT_DIR if IS_KAGGLE else "./anomalib_results"),
        models=models,
        categories=categories,
        precision=precision,
    )
    return train_all(args)


# =====================================================================
# Main (CLI 入口)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Anomalib 訓練 Baseline 模型 (PatchCore, CFlow, RD4AD, EfficientAD)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=KAGGLE_DATA_ROOT if IS_KAGGLE else "./mvtec",
        help="MVTec 資料集根目錄 (包含各類別資料夾)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=KAGGLE_OUTPUT_DIR if IS_KAGGLE else "./anomalib_results",
        help="訓練結果與 checkpoint 輸出目錄",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="要訓練的模型 (預設全部): PatchCore CFlow RD4AD EfficientAD",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="要訓練的 MVTec 類別 (預設全部15類)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="訓練精度 (e.g., 16-mixed, 32)",
    )

    args = parser.parse_args()
    train_all(args)
