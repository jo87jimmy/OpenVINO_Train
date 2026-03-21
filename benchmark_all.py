#!/usr/bin/env python
"""
統一 Benchmark：比較所有異常檢測模型在 MVTec 資料集上的表現。

模型:
  1. PatchCore   (Anomalib)
  2. CFlow       (Anomalib)   ← 對應 "CFRG"
  3. RD4AD       (Anomalib)   ← Reverse Distillation
  4. EfficientAD (Anomalib)
  5. DRAEM       (Teacher)    ← 自訂模型
  6. Ours        (Student)    ← 自訂壓縮模型

指標:
  - Image-AUROC, Pixel-AUROC
  - Image-AP,    Pixel-AP
  - Inference Time (ms), FPS

Usage:
    python benchmark_all.py --data_root ./data-mvtec/mvtec --obj_id -1
    python benchmark_all.py --data_root ./data-mvtec/mvtec --categories bottle carpet
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── 自訂模型 ──
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEM_Test_Visual_Dataset

# ── Anomalib ──
from anomalib.data import MVTec
from anomalib.models import Patchcore, Cflow, ReverseDistillation, EfficientAd
from anomalib.engine import Engine

# =====================================================================
# 常數
# =====================================================================
MVTEC_CATEGORIES = [
    "capsule", "bottle", "carpet", "leather", "pill",
    "transistor", "tile", "cable", "zipper", "toothbrush",
    "metal_nut", "hazelnut", "screw", "grid", "wood",
]

IMG_SIZE = 256

# 模型顯示順序與顏色
MODEL_DISPLAY = {
    "PatchCore":   "#1f77b4",
    "CFlow":       "#ff7f0e",
    "RD4AD":       "#2ca02c",
    "EfficientAD": "#9467bd",
    "DRAEM":       "#e74c3c",
    "Ours":        "#17becf",
}


# =====================================================================
# 工具函數
# =====================================================================
def setup_seed(seed=111):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=-2):
    if gpu_id == -1 or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id == -2:
        # 自動選擇記憶體最少的 GPU
        mem = []
        for i in range(torch.cuda.device_count()):
            mem.append((i, torch.cuda.memory_allocated(i)))
        gpu_id = min(mem, key=lambda x: x[1])[0]
    return torch.device(f"cuda:{gpu_id}")


# =====================================================================
# 模型包裝器：統一推論介面
# =====================================================================
class DRAEMWrapper:
    """包裝 DRAEM 系列模型 (Teacher / Student)。"""

    def __init__(self, name, recon_path, seg_path, base_width, base_channels, device):
        self.name = name
        self.device = device

        self.recon = ReconstructiveSubNetwork(
            in_channels=3, out_channels=3, base_width=base_width
        )
        self.seg = DiscriminativeSubNetwork(
            in_channels=6, out_channels=2, base_channels=base_channels
        )

        self.recon.load_state_dict(torch.load(recon_path, map_location=device))
        self.seg.load_state_dict(torch.load(seg_path, map_location=device))

        self.recon.to(device).eval()
        self.seg.to(device).eval()

    def predict(self, image_tensor):
        """
        輸入: image_tensor (B, 3, H, W)
        輸出: (image_scores, anomaly_maps)
              image_scores: (B,) 每張圖的異常分數
              anomaly_maps: (B, H, W) 像素級異常熱圖
        """
        with torch.no_grad():
            recon = self.recon(image_tensor)
            joined = torch.cat((recon, image_tensor), dim=1)
            out = self.seg(joined)
            prob = torch.softmax(out, dim=1)[:, 1, :, :]  # (B, H, W) — 異常機率
            image_scores = prob.reshape(prob.shape[0], -1).max(dim=1).values  # (B,)
        return image_scores.cpu().numpy(), prob.cpu().numpy()

    def count_params(self):
        p1 = sum(p.numel() for p in self.recon.parameters())
        p2 = sum(p.numel() for p in self.seg.parameters())
        return p1 + p2


class AnomalibWrapper:
    """包裝 Anomalib 模型，提供統一推論介面。"""

    def __init__(self, name, model_class, ckpt_path, data_root, category, device):
        self.name = name
        self.device = device
        self.model_class = model_class
        self.ckpt_path = ckpt_path
        self.data_root = data_root
        self.category = category

        # 載入模型
        self.model = model_class.load_from_checkpoint(ckpt_path)
        self.model.to(device).eval()

    def predict(self, image_tensor):
        """
        透過模型前向傳播取得預測。
        anomalib 模型在 eval 模式下的 forward 會回傳包含 anomaly_map 和 pred_score 的結果。
        """
        with torch.no_grad():
            # anomalib v1.x: model 在 eval 模式下接受 batch tensor
            # 回傳 InferenceBatch 或 dict
            output = self.model(image_tensor)

        # 嘗試從不同格式的輸出中取得 anomaly_map 和 pred_score
        anomaly_maps, image_scores = self._extract_predictions(output, image_tensor)
        return image_scores, anomaly_maps

    def _extract_predictions(self, output, image_tensor):
        """從 anomalib 輸出中提取預測結果。"""
        B, _, H, W = image_tensor.shape

        # 嘗試多種輸出格式 (相容不同 anomalib 版本)
        anomaly_map = None
        pred_score = None

        if hasattr(output, "anomaly_map"):
            anomaly_map = output.anomaly_map
        elif isinstance(output, dict) and "anomaly_map" in output:
            anomaly_map = output["anomaly_map"]

        if hasattr(output, "pred_score"):
            pred_score = output.pred_score
        elif isinstance(output, dict) and "pred_score" in output:
            pred_score = output["pred_score"]

        # 處理 anomaly_map
        if anomaly_map is not None:
            if isinstance(anomaly_map, torch.Tensor):
                anomaly_map = anomaly_map.cpu().numpy()
            # 確保形狀為 (B, H, W)
            if anomaly_map.ndim == 4:
                anomaly_map = anomaly_map.squeeze(1)
            # 如果尺寸不匹配，resize
            if anomaly_map.shape[-2:] != (H, W):
                am_tensor = torch.from_numpy(anomaly_map).unsqueeze(1).float()
                am_tensor = F.interpolate(am_tensor, size=(H, W), mode="bilinear", align_corners=False)
                anomaly_map = am_tensor.squeeze(1).numpy()
        else:
            # fallback: 用零圖代替
            anomaly_map = np.zeros((B, H, W), dtype=np.float32)

        # 處理 pred_score
        if pred_score is not None:
            if isinstance(pred_score, torch.Tensor):
                pred_score = pred_score.cpu().numpy()
            pred_score = pred_score.flatten()
        else:
            # fallback: 從 anomaly_map 取最大值
            pred_score = anomaly_map.reshape(B, -1).max(axis=1)

        return anomaly_map, pred_score

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())


# =====================================================================
# 指標計算
# =====================================================================
def compute_metrics(all_labels, all_scores, all_masks, all_anomaly_maps):
    """
    計算所有指標。

    Args:
        all_labels:       list of int (0/1)，每張圖的 GT 標籤
        all_scores:       list of float，每張圖的異常分數
        all_masks:        list of ndarray (H, W)，每張圖的 GT mask (0/1)
        all_anomaly_maps: list of ndarray (H, W)，每張圖的預測異常熱圖

    Returns:
        dict with Image-AUROC, Pixel-AUROC, Image-AP, Pixel-AP
    """
    labels = np.array(all_labels)
    scores = np.array(all_scores)

    metrics = {}

    # ── Image-level ──
    try:
        metrics["Image-AUROC"] = roc_auc_score(labels, scores)
    except ValueError:
        metrics["Image-AUROC"] = 0.0

    try:
        metrics["Image-AP"] = average_precision_score(labels, scores)
    except ValueError:
        metrics["Image-AP"] = 0.0

    # ── Pixel-level ──
    gt_pixels = np.concatenate([m.flatten() for m in all_masks])
    pred_pixels = np.concatenate([a.flatten() for a in all_anomaly_maps])

    # 確保 GT 有正例和負例
    if gt_pixels.max() > 0 and gt_pixels.min() == 0:
        gt_binary = (gt_pixels > 0.5).astype(np.int32)
        try:
            metrics["Pixel-AUROC"] = roc_auc_score(gt_binary, pred_pixels)
        except ValueError:
            metrics["Pixel-AUROC"] = 0.0
        try:
            metrics["Pixel-AP"] = average_precision_score(gt_binary, pred_pixels)
        except ValueError:
            metrics["Pixel-AP"] = 0.0
    else:
        metrics["Pixel-AUROC"] = 0.0
        metrics["Pixel-AP"] = 0.0

    return metrics


# =====================================================================
# 推論 Benchmark（計算指標 + 速度）
# =====================================================================
def benchmark_model_draem(wrapper, data_root, category, device, n_repeat=3):
    """
    對 DRAEM 系列模型 (Teacher/Student) 進行 benchmark。
    使用自訂 DataLoader。
    """
    test_path = os.path.join(data_root, category, "test")
    if not os.path.exists(test_path):
        print(f"    ⚠️ 測試資料不存在: {test_path}")
        return None

    dataset = MVTecDRAEM_Test_Visual_Dataset(test_path, resize_shape=[IMG_SIZE, IMG_SIZE])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── 暖機 ──
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(5):
        wrapper.predict(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── 推論 + 收集預測 ──
    all_labels = []
    all_scores = []
    all_masks = []
    all_anomaly_maps = []
    all_times = []

    for repeat in range(n_repeat):
        for sample in dataloader:
            image = sample["image"].to(device)
            label = sample["has_anomaly"].numpy().flatten()[0]
            mask = sample["mask"].numpy().squeeze()  # (H, W)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            img_score, anom_map = wrapper.predict(image)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            all_times.append(elapsed_ms)

            # 只在第一次 repeat 收集指標資料
            if repeat == 0:
                all_labels.append(int(label > 0.5))
                all_scores.append(float(img_score[0]))
                all_masks.append(mask)
                all_anomaly_maps.append(anom_map[0])  # (H, W)

    # ── 計算指標 ──
    metrics = compute_metrics(all_labels, all_scores, all_masks, all_anomaly_maps)

    # ── 速度指標 ──
    avg_time_ms = np.mean(all_times)
    std_time_ms = np.std(all_times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    metrics["Inference Time (ms)"] = avg_time_ms
    metrics["Inference Std (ms)"] = std_time_ms
    metrics["FPS"] = fps
    metrics["Params"] = wrapper.count_params()

    return metrics


def benchmark_model_anomalib(wrapper, data_root, category, device, n_repeat=3):
    """
    對 Anomalib 模型進行 benchmark。
    使用自訂 DataLoader（與 DRAEM 相同，確保資料一致性）。
    """
    test_path = os.path.join(data_root, category, "test")
    if not os.path.exists(test_path):
        print(f"    ⚠️ 測試資料不存在: {test_path}")
        return None

    dataset = MVTecDRAEM_Test_Visual_Dataset(test_path, resize_shape=[IMG_SIZE, IMG_SIZE])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── 暖機 ──
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(5):
        wrapper.predict(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── 推論 + 收集預測 ──
    all_labels = []
    all_scores = []
    all_masks = []
    all_anomaly_maps = []
    all_times = []

    for repeat in range(n_repeat):
        for sample in dataloader:
            image = sample["image"].to(device)
            label = sample["has_anomaly"].numpy().flatten()[0]
            mask = sample["mask"].numpy().squeeze()  # (H, W)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            img_score, anom_map = wrapper.predict(image)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            all_times.append(elapsed_ms)

            if repeat == 0:
                all_labels.append(int(label > 0.5))
                all_scores.append(float(img_score[0]))
                all_masks.append(mask)
                all_anomaly_maps.append(anom_map[0])

    metrics = compute_metrics(all_labels, all_scores, all_masks, all_anomaly_maps)

    avg_time_ms = np.mean(all_times)
    std_time_ms = np.std(all_times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    metrics["Inference Time (ms)"] = avg_time_ms
    metrics["Inference Std (ms)"] = std_time_ms
    metrics["FPS"] = fps
    metrics["Params"] = wrapper.count_params()

    return metrics


# =====================================================================
# 載入所有模型
# =====================================================================
def load_models(args, category, device):
    """載入所有要比較的模型，回傳 (model_name, wrapper, benchmark_fn) 列表。"""
    models = []

    # ── 1. Anomalib 模型 ──
    anomalib_models = {
        "PatchCore": Patchcore,
        "CFlow": Cflow,
        "RD4AD": ReverseDistillation,
        "EfficientAD": EfficientAd,
    }

    # 讀取 checkpoint registry
    registry_path = os.path.join(args.anomalib_dir, "checkpoint_registry.json")
    ckpt_registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            ckpt_registry = json.load(f)

    for model_name, model_class in anomalib_models.items():
        # 嘗試從 registry 取得 checkpoint
        ckpt_path = None
        if model_name in ckpt_registry and category in ckpt_registry[model_name]:
            ckpt_path = ckpt_registry[model_name][category]

        # fallback: 搜尋標準路徑
        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            standard_path = os.path.join(
                args.anomalib_dir, "checkpoints", model_name, f"{category}.ckpt"
            )
            if os.path.exists(standard_path):
                ckpt_path = standard_path

        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            print(f"    ⚠️ 跳過 {model_name}: checkpoint 不存在 ({category})")
            continue

        try:
            wrapper = AnomalibWrapper(
                name=model_name,
                model_class=model_class,
                ckpt_path=ckpt_path,
                data_root=args.data_root,
                category=category,
                device=device,
            )
            models.append((model_name, wrapper, benchmark_model_anomalib))
            print(f"    ✅ {model_name} 載入成功")
        except Exception as e:
            print(f"    ❌ {model_name} 載入失敗: {e}")

    # ── 2. DRAEM Teacher ──
    draem_recon_path = os.path.join(
        args.draem_dir,
        f"DRAEM_seg_large_ae_large_0.0001_800_bs8_{category}_.pckl",
    )
    draem_seg_path = os.path.join(
        args.draem_dir,
        f"DRAEM_seg_large_ae_large_0.0001_800_bs8_{category}__seg.pckl",
    )
    if os.path.exists(draem_recon_path) and os.path.exists(draem_seg_path):
        try:
            wrapper = DRAEMWrapper(
                name="DRAEM",
                recon_path=draem_recon_path,
                seg_path=draem_seg_path,
                base_width=128,
                base_channels=64,
                device=device,
            )
            models.append(("DRAEM", wrapper, benchmark_model_draem))
            print(f"    ✅ DRAEM (Teacher) 載入成功")
        except Exception as e:
            print(f"    ❌ DRAEM (Teacher) 載入失敗: {e}")
    else:
        print(f"    ⚠️ 跳過 DRAEM (Teacher): checkpoint 不存在")

    # ── 3. Ours (Student) ──
    student_recon_path = os.path.join(
        args.student_dir, f"{category}_best_recon.pckl"
    )
    student_seg_path = os.path.join(
        args.student_dir, f"{category}_best_seg.pckl"
    )
    if os.path.exists(student_recon_path) and os.path.exists(student_seg_path):
        try:
            wrapper = DRAEMWrapper(
                name="Ours",
                recon_path=student_recon_path,
                seg_path=student_seg_path,
                base_width=64,
                base_channels=32,
                device=device,
            )
            models.append(("Ours", wrapper, benchmark_model_draem))
            print(f"    ✅ Ours (Student) 載入成功")
        except Exception as e:
            print(f"    ❌ Ours (Student) 載入失敗: {e}")
    else:
        print(f"    ⚠️ 跳過 Ours (Student): checkpoint 不存在")

    return models


# =====================================================================
# 視覺化
# =====================================================================
def plot_metric_comparison(all_results, metric_name, save_path, higher_is_better=True):
    """繪製單一指標的跨模型比較圖。"""
    categories = list(all_results.keys())
    model_names = set()
    for cat_results in all_results.values():
        model_names.update(cat_results.keys())
    model_names = [m for m in MODEL_DISPLAY.keys() if m in model_names]

    if not model_names or not categories:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(categories) * 1.2), 7))

    x = np.arange(len(categories))
    width = 0.8 / len(model_names)

    for i, model_name in enumerate(model_names):
        values = []
        for cat in categories:
            if model_name in all_results[cat] and all_results[cat][model_name] is not None:
                values.append(all_results[cat][model_name].get(metric_name, 0.0))
            else:
                values.append(0.0)

        color = MODEL_DISPLAY.get(model_name, "#333333")
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

    direction = "Higher is Better ↑" if higher_is_better else "Lower is Better ↓"
    ax.set_title(f"{metric_name} Comparison ({direction})", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_table(all_results, save_path):
    """繪製所有指標的平均值摘要表格。"""
    model_names = set()
    for cat_results in all_results.values():
        model_names.update(cat_results.keys())
    model_names = [m for m in MODEL_DISPLAY.keys() if m in model_names]

    if not model_names:
        return

    metric_keys = ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP", "Inference Time (ms)", "FPS"]

    # 計算每個模型的平均指標
    avg_data = {}
    for model_name in model_names:
        avg_data[model_name] = {}
        for metric in metric_keys:
            values = []
            for cat_results in all_results.values():
                if model_name in cat_results and cat_results[model_name] is not None:
                    val = cat_results[model_name].get(metric, None)
                    if val is not None:
                        values.append(val)
            avg_data[model_name][metric] = np.mean(values) if values else 0.0

    # 繪製表格
    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 2.5), 5))
    ax.axis("off")
    ax.set_title("Average Benchmark Results Across All Categories", fontsize=14, fontweight="bold", pad=20)

    col_labels = ["Method"] + metric_keys
    table_data = []
    cell_colors = []

    for model_name in model_names:
        row = [model_name]
        row_colors = [MODEL_DISPLAY.get(model_name, "#ffffff") + "40"]  # 淡色背景
        for metric in metric_keys:
            val = avg_data[model_name][metric]
            if "AUROC" in metric or "AP" in metric:
                row.append(f"{val:.4f}")
            elif "Time" in metric:
                row.append(f"{val:.2f}")
            elif "FPS" in metric:
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.4f}")
            row_colors.append("#ffffff")
        table_data.append(row)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # 標題列加粗
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#d4e6f1")

    # 高亮最佳值
    for col_idx, metric in enumerate(metric_keys, start=1):
        vals = []
        for row_idx, model_name in enumerate(model_names):
            vals.append((row_idx, avg_data[model_name][metric]))

        if not vals:
            continue

        # 決定最佳方向
        if "Time" in metric:
            best_idx = min(vals, key=lambda x: x[1])[0]
        else:
            best_idx = max(vals, key=lambda x: x[1])[0]

        table[best_idx + 1, col_idx].set_facecolor("#abebc6")
        table[best_idx + 1, col_idx].set_text_props(fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_csv_report(all_results, save_path):
    """輸出 CSV 格式的完整結果。"""
    model_names = set()
    for cat_results in all_results.values():
        model_names.update(cat_results.keys())
    model_names = sorted(model_names)

    metric_keys = ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP",
                    "Inference Time (ms)", "Inference Std (ms)", "FPS", "Params"]

    with open(save_path, "w") as f:
        # Header
        f.write("Category,Method," + ",".join(metric_keys) + "\n")

        for category in sorted(all_results.keys()):
            for model_name in model_names:
                if model_name in all_results[category] and all_results[category][model_name] is not None:
                    m = all_results[category][model_name]
                    vals = [str(m.get(k, "N/A")) for k in metric_keys]
                    f.write(f"{category},{model_name},{','.join(vals)}\n")

        # 平均值
        f.write("\n")
        for model_name in model_names:
            avg_vals = []
            for k in metric_keys:
                values = []
                for cat_results in all_results.values():
                    if model_name in cat_results and cat_results[model_name] is not None:
                        val = cat_results[model_name].get(k, None)
                        if val is not None:
                            values.append(val)
                avg_vals.append(str(np.mean(values)) if values else "N/A")
            f.write(f"AVERAGE,{model_name},{','.join(avg_vals)}\n")


# =====================================================================
# 主函數
# =====================================================================
def main(args):
    setup_seed(111)
    device = get_device(args.gpu_id)

    categories = args.categories if args.categories else MVTEC_CATEGORIES

    save_root = args.save_dir
    os.makedirs(save_root, exist_ok=True)

    print("=" * 80)
    print("  統一 Benchmark：所有異常檢測模型在 MVTec 上的綜合比較")
    print("=" * 80)
    print(f"  類別數: {len(categories)}")
    print(f"  裝置: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  推論重複次數: {args.n_repeat}")
    print(f"  影像大小: {IMG_SIZE}×{IMG_SIZE}")
    print("=" * 80)

    # 收集所有結果: {category: {model_name: metrics_dict}}
    all_results = {}

    for category in categories:
        print(f"\n{'━' * 70}")
        print(f"  類別: {category}")
        print(f"{'━' * 70}")

        # 載入所有模型
        models = load_models(args, category, device)

        if not models:
            print(f"  ⚠️ 沒有可用模型，跳過 {category}")
            continue

        all_results[category] = {}

        for model_name, wrapper, bench_fn in models:
            print(f"\n  ⏱️  Benchmark: {model_name}")
            try:
                metrics = bench_fn(wrapper, args.data_root, category, device, args.n_repeat)
            except Exception as e:
                print(f"    ❌ Benchmark 失敗: {e}")
                metrics = None

            if metrics:
                all_results[category][model_name] = metrics
                print(f"    Image-AUROC: {metrics['Image-AUROC']:.4f}  |  "
                      f"Pixel-AUROC: {metrics['Pixel-AUROC']:.4f}  |  "
                      f"Image-AP: {metrics['Image-AP']:.4f}  |  "
                      f"Pixel-AP: {metrics['Pixel-AP']:.4f}")
                print(f"    Inference: {metrics['Inference Time (ms)']:.2f} ms  |  "
                      f"FPS: {metrics['FPS']:.1f}  |  "
                      f"Params: {metrics['Params']:,}")

        # 釋放記憶體
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ================================================================
    # 結果彙整
    # ================================================================
    if not all_results:
        print("\n❌ 沒有成功的 benchmark 結果")
        return

    print(f"\n\n{'=' * 80}")
    print("  結果彙整")
    print(f"{'=' * 80}")

    # ── 印出摘要表 ──
    model_names = set()
    for cat_results in all_results.values():
        model_names.update(cat_results.keys())
    model_names = [m for m in MODEL_DISPLAY.keys() if m in model_names]

    metric_keys = ["Image-AUROC", "Pixel-AUROC", "Image-AP", "Pixel-AP", "Inference Time (ms)", "FPS"]

    # 每類別的詳細表
    for category in sorted(all_results.keys()):
        print(f"\n  ── {category} ──")
        header = f"  {'Method':<15}" + "".join(f"{k:>18}" for k in metric_keys)
        print(header)
        print(f"  {'─' * (15 + 18 * len(metric_keys))}")
        for model_name in model_names:
            if model_name in all_results[category]:
                m = all_results[category][model_name]
                row = f"  {model_name:<15}"
                for k in metric_keys:
                    val = m.get(k, 0.0)
                    if "AUROC" in k or "AP" in k:
                        row += f"{val:>18.4f}"
                    elif "Time" in k:
                        row += f"{val:>18.2f}"
                    elif "FPS" in k:
                        row += f"{val:>18.1f}"
                print(row)

    # 平均值表
    print(f"\n  ── AVERAGE ──")
    header = f"  {'Method':<15}" + "".join(f"{k:>18}" for k in metric_keys)
    print(header)
    print(f"  {'─' * (15 + 18 * len(metric_keys))}")

    for model_name in model_names:
        row = f"  {model_name:<15}"
        for k in metric_keys:
            values = []
            for cat_results in all_results.values():
                if model_name in cat_results and cat_results[model_name] is not None:
                    val = cat_results[model_name].get(k, None)
                    if val is not None:
                        values.append(val)
            avg = np.mean(values) if values else 0.0
            if "AUROC" in k or "AP" in k:
                row += f"{avg:>18.4f}"
            elif "Time" in k:
                row += f"{avg:>18.2f}"
            elif "FPS" in k:
                row += f"{avg:>18.1f}"
        print(row)

    # ── 生成圖表 ──
    print(f"\n  📊 生成比較圖表...")

    plot_metric_comparison(all_results, "Image-AUROC",
                           os.path.join(save_root, "comparison_image_auroc.png"), higher_is_better=True)
    plot_metric_comparison(all_results, "Pixel-AUROC",
                           os.path.join(save_root, "comparison_pixel_auroc.png"), higher_is_better=True)
    plot_metric_comparison(all_results, "Image-AP",
                           os.path.join(save_root, "comparison_image_ap.png"), higher_is_better=True)
    plot_metric_comparison(all_results, "Pixel-AP",
                           os.path.join(save_root, "comparison_pixel_ap.png"), higher_is_better=True)
    plot_metric_comparison(all_results, "Inference Time (ms)",
                           os.path.join(save_root, "comparison_inference_time.png"), higher_is_better=False)
    plot_metric_comparison(all_results, "FPS",
                           os.path.join(save_root, "comparison_fps.png"), higher_is_better=True)

    plot_summary_table(all_results, os.path.join(save_root, "summary_table.png"))

    # ── CSV 報告 ──
    csv_path = os.path.join(save_root, "benchmark_results.csv")
    generate_csv_report(all_results, csv_path)
    print(f"  📄 CSV 報告: {csv_path}")

    # ── JSON 結果 ──
    json_path = os.path.join(save_root, "benchmark_results.json")
    # 轉換 numpy 為 Python 原生型別
    json_safe = {}
    for cat, cat_results in all_results.items():
        json_safe[cat] = {}
        for model, metrics in cat_results.items():
            if metrics is not None:
                json_safe[cat][model] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                                         for k, v in metrics.items()}
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"  📄 JSON 結果: {json_path}")

    print(f"\n{'=' * 80}")
    print(f"  🎉 Benchmark 完成！結果儲存於: {os.path.abspath(save_root)}")
    print(f"{'=' * 80}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="統一 Benchmark：比較所有異常檢測模型在 MVTec 上的綜合表現"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data-mvtec/mvtec",
        help="MVTec 資料集根目錄",
    )
    parser.add_argument(
        "--anomalib_dir", type=str, default="./anomalib_results",
        help="Anomalib 訓練結果目錄 (含 checkpoint_registry.json)",
    )
    parser.add_argument(
        "--draem_dir", type=str, default="./DRAEM_checkpoints",
        help="DRAEM Teacher checkpoint 目錄",
    )
    parser.add_argument(
        "--student_dir", type=str, default="./student_model_checkpoints",
        help="Student (Ours) checkpoint 目錄",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./benchmark_results",
        help="結果儲存目錄",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="指定 MVTec 類別 (預設全部15類)",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-2,
        help="GPU ID (-2: auto, -1: CPU)",
    )
    parser.add_argument(
        "--n_repeat", type=int, default=3,
        help="推論重複次數 (取平均，預設 3)",
    )

    args = parser.parse_args()
    main(args)
