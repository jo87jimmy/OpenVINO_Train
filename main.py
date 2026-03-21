import os
import sys
import time
import torch
import numpy as np
import random
import argparse
from data_loader import MVTecDRAEM_Test_Visual_Dataset
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 非互動式後端，避免彈出視窗


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        gpu_memory.append((i, memory_allocated))
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


# =======================
# 模型效率指標工具函數
# =======================
def count_parameters(model):
    """計算模型的總參數量與可訓練參數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """計算模型大小 (MB)"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024**2)
    return total_size_mb


def measure_gpu_memory(model, input_tensor):
    """測量模型推論時的 GPU 記憶體使用量"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    mem_after = torch.cuda.max_memory_allocated()
    return (mem_after - mem_before) / (1024**2)  # MB


def warm_up_model(model, input_tensor, n_warmup=10):
    """GPU 暖機，避免首次推論偏慢"""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()


# =======================
# 推論時間測量（單一模型完整 pipeline）
# =======================
def benchmark_inference(recon_model, seg_model, dataloader, n_repeat=1, label="Model"):
    """
    測量完整推論 pipeline (reconstruction + segmentation) 的時間。
    回傳: 每張圖平均時間(ms), 總時間(s), FPS, 各圖片時間列表
    """
    recon_model.eval()
    seg_model.eval()

    all_times = []

    with torch.no_grad():
        for repeat_idx in range(n_repeat):
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()

                torch.cuda.synchronize()
                start = time.perf_counter()

                # Stage 1: 重建
                gray_rec = recon_model(gray_batch)
                # Stage 2: 分割
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
                out_mask = seg_model(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                torch.cuda.synchronize()
                end = time.perf_counter()

                elapsed_ms = (end - start) * 1000.0
                all_times.append(elapsed_ms)

    total_images = len(all_times)
    total_time_s = sum(all_times) / 1000.0
    avg_time_ms = np.mean(all_times)
    std_time_ms = np.std(all_times)
    fps = total_images / total_time_s if total_time_s > 0 else 0

    return {
        "label": label,
        "total_images": total_images,
        "avg_time_ms": avg_time_ms,
        "std_time_ms": std_time_ms,
        "total_time_s": total_time_s,
        "fps": fps,
        "all_times": all_times,
    }


def benchmark_recon_only(recon_model, dataloader, n_repeat=1, label="Recon"):
    """只測量重建模型的推論時間"""
    recon_model.eval()
    all_times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = recon_model(gray_batch)
                torch.cuda.synchronize()
                end = time.perf_counter()
                all_times.append((end - start) * 1000.0)
    avg_ms = np.mean(all_times)
    std_ms = np.std(all_times)
    fps = len(all_times) / (sum(all_times) / 1000.0) if sum(all_times) > 0 else 0
    return {"label": label, "avg_time_ms": avg_ms, "std_time_ms": std_ms, "fps": fps}


def benchmark_seg_only(recon_model, seg_model, dataloader, n_repeat=1, label="Seg"):
    """只測量分割模型的推論時間（需先通過重建模型取得輸入）"""
    recon_model.eval()
    seg_model.eval()
    all_times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                gray_rec = recon_model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_mask = seg_model(joined_in)
                _ = torch.softmax(out_mask, dim=1)
                torch.cuda.synchronize()
                end = time.perf_counter()
                all_times.append((end - start) * 1000.0)
    avg_ms = np.mean(all_times)
    std_ms = np.std(all_times)
    fps = len(all_times) / (sum(all_times) / 1000.0) if sum(all_times) > 0 else 0
    return {"label": label, "avg_time_ms": avg_ms, "std_time_ms": std_ms, "fps": fps}


# =======================
# 視覺化比較圖表
# =======================
def plot_comparison(teacher_result, student_result, save_dir, obj_name):
    """生成教師 vs 學生模型的比較圖表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Teacher vs Student Model Inference Efficiency Comparison — {obj_name}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. 平均推論時間比較 (Bar chart)
    ax = axes[0, 0]
    labels = ["Teacher (origin)", "Student"]
    avg_times = [teacher_result["avg_time_ms"], student_result["avg_time_ms"]]
    std_times = [teacher_result["std_time_ms"], student_result["std_time_ms"]]
    bars = ax.bar(
        labels,
        avg_times,
        yerr=std_times,
        capsize=5,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_ylabel("Avg Inference Time (ms)")
    ax.set_title("Avg Inference Time (Lower is Better)")
    for bar, val in zip(bars, avg_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. FPS 比較 (Bar chart)
    ax = axes[0, 1]
    fps_vals = [teacher_result["fps"], student_result["fps"]]
    bars = ax.bar(
        labels, fps_vals, color=["#e74c3c", "#2ecc71"], edgecolor="black", alpha=0.85
    )
    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title("Inference Speed FPS (Higher is Better)")
    for bar, val in zip(bars, fps_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. 逐張推論時間分佈 (Histogram)
    ax = axes[1, 0]
    ax.hist(
        teacher_result["all_times"],
        bins=30,
        alpha=0.6,
        label="Teacher",
        color="#e74c3c",
    )
    ax.hist(
        student_result["all_times"],
        bins=30,
        alpha=0.6,
        label="Student",
        color="#2ecc71",
    )
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Time Distribution")
    ax.legend()

    # 4. 加速比與壓縮比資訊表
    ax = axes[1, 1]
    ax.axis("off")
    speedup = (
        teacher_result["avg_time_ms"] / student_result["avg_time_ms"]
        if student_result["avg_time_ms"] > 0
        else float("inf")
    )
    fps_ratio = (
        student_result["fps"] / teacher_result["fps"]
        if teacher_result["fps"] > 0
        else float("inf")
    )

    info_text = (
        f"{'Metric':<28}{'Teacher (origin)':<18}{'Student':<18}{'Ratio':<12}\n"
        f"{'─' * 76}\n"
        f"{'Avg Inference Time (ms)':<28}{teacher_result['avg_time_ms']:<18.2f}{student_result['avg_time_ms']:<18.2f}{speedup:<12.2f}\n"
        f"{'Std Dev (ms)':<28}{teacher_result['std_time_ms']:<18.2f}{student_result['std_time_ms']:<18.2f}\n"
        f"{'FPS':<28}{teacher_result['fps']:<18.1f}{student_result['fps']:<18.1f}{fps_ratio:<12.2f}\n"
        f"{'Total Inference Time (s)':<28}{teacher_result['total_time_s']:<18.3f}{student_result['total_time_s']:<18.3f}\n"
        f"{'Total Images':<28}{teacher_result['total_images']:<18}{student_result['total_images']:<18}\n"
    )
    ax.text(
        0.05,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{obj_name}_inference_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 比較圖表已儲存: {save_path}")
    return save_path


def plot_model_params_comparison(teacher_info, student_info, save_dir, obj_name):
    """生成模型參數量 / 大小比較圖"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Model Structure Comparison — {obj_name}", fontsize=14, fontweight="bold"
    )

    labels = ["Teacher (origin)", "Student"]
    colors = ["#e74c3c", "#2ecc71"]

    # 1. 總參數量
    ax = axes[0]
    params = [teacher_info["total_params"] / 1e6, student_info["total_params"] / 1e6]
    bars = ax.bar(labels, params, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Total Model Parameters")
    for bar, val in zip(bars, params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. 模型大小 (MB)
    ax = axes[1]
    sizes = [teacher_info["model_size_mb"], student_info["model_size_mb"]]
    bars = ax.bar(labels, sizes, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size")
    for bar, val in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. GPU 記憶體使用
    ax = axes[2]
    gpu_mem = [teacher_info["gpu_mem_mb"], student_info["gpu_mem_mb"]]
    bars = ax.bar(labels, gpu_mem, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("Inference GPU Memory Usage")
    for bar, val in zip(bars, gpu_mem):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{obj_name}_model_structure_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 模型結構比較圖已儲存: {save_path}")
    return save_path


def plot_stage_breakdown(
    teacher_recon, teacher_seg, student_recon, student_seg, save_dir, obj_name
):
    """生成重建 vs 分割階段時間拆解比較圖"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Inference Stage Time Breakdown — {obj_name}", fontsize=14, fontweight="bold"
    )

    x = np.arange(2)
    width = 0.35

    recon_times = [teacher_recon["avg_time_ms"], student_recon["avg_time_ms"]]
    seg_times = [teacher_seg["avg_time_ms"], student_seg["avg_time_ms"]]

    bars1 = ax.bar(
        x - width / 2,
        recon_times,
        width,
        label="Reconstruction",
        color="#3498db",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        seg_times,
        width,
        label="Segmentation",
        color="#f39c12",
        alpha=0.85,
    )

    ax.set_ylabel("Avg Inference Time (ms)")
    ax.set_title("Avg Inference Time per Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(["Teacher (origin)", "Student"])
    ax.legend()

    for bar, val in zip(bars1, recon_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, seg_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{obj_name}_stage_breakdown.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 階段拆解圖已儲存: {save_path}")
    return save_path


# =======================
# 主函數
# =======================
def main(obj_names, args):
    setup_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_root = "./comparison_results"
    os.makedirs(save_root, exist_ok=True)

    n_repeat = args.n_repeat  # 重複推論次數（取平均更穩定）

    print("=" * 80)
    print("  教師模型 vs 學生模型 — 推論時間與效率指標比較")
    print("=" * 80)
    print(f"  重複推論次數: {n_repeat}")
    print(f"  物件類別數: {len(obj_names)}")
    print(f"  裝置: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 80)

    all_results = []

    for obj_name in obj_names:
        print(f"\n{'─' * 60}")
        print(f"  物件類別: {obj_name}")
        print(f"{'─' * 60}")

        img_dim = 256

        # =====================
        # 載入教師模型 (Teacher)
        # =====================
        teacher_recon = ReconstructiveSubNetwork(
            in_channels=3, out_channels=3, base_width=128
        )
        teacher_recon_path = (
            "./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_"
            + obj_name
            + "_.pckl"
        )
        if not os.path.exists(teacher_recon_path):
            print(f"  ❌ 教師重建模型權重未找到: {teacher_recon_path}")
            continue
        teacher_recon.load_state_dict(
            torch.load(teacher_recon_path, map_location=device)
        )
        teacher_recon.cuda()
        teacher_recon.eval()

        teacher_seg = DiscriminativeSubNetwork(
            in_channels=6, out_channels=2, base_channels=64
        )
        teacher_seg_path = (
            "./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_"
            + obj_name
            + "__seg.pckl"
        )
        if not os.path.exists(teacher_seg_path):
            print(f"  ❌ 教師分割模型權重未找到: {teacher_seg_path}")
            continue
        teacher_seg.load_state_dict(torch.load(teacher_seg_path, map_location=device))
        teacher_seg.cuda()
        teacher_seg.eval()

        print("  ✅ 教師模型載入完成 (Recon base_width=128, Seg base_channels=64)")

        # =====================
        # 載入學生模型 (Student)
        # =====================
        student_recon = ReconstructiveSubNetwork(
            in_channels=3, out_channels=3, base_width=64
        )
        student_recon_path = (
            "./student_model_checkpoints/" + obj_name + "_best_recon.pckl"
        )
        if not os.path.exists(student_recon_path):
            print(f"  ❌ 學生重建模型權重未找到: {student_recon_path}")
            continue
        student_recon.load_state_dict(
            torch.load(student_recon_path, map_location=device)
        )
        student_recon.cuda()
        student_recon.eval()

        student_seg_m = DiscriminativeSubNetwork(
            in_channels=6, out_channels=2, base_channels=32
        )
        student_seg_path = "./student_model_checkpoints/" + obj_name + "_best_seg.pckl"
        if not os.path.exists(student_seg_path):
            print(f"  ❌ 學生分割模型權重未找到: {student_seg_path}")
            continue
        student_seg_m.load_state_dict(torch.load(student_seg_path, map_location=device))
        student_seg_m.cuda()
        student_seg_m.eval()

        print("  ✅ 學生模型載入完成 (Recon base_width=64, Seg base_channels=32)")

        # =====================
        # 載入資料集
        # =====================
        path = args.mvtec_root + "/" + obj_name + "/test/"
        if not os.path.exists(path):
            print(f"  ❌ 資料集路徑不存在: {path}")
            continue

        dataset = MVTecDRAEM_Test_Visual_Dataset(path, resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"  📂 資料集大小: {len(dataset)} 張圖片")

        # =====================
        # 模型結構指標
        # =====================
        dummy_input = torch.randn(1, 3, img_dim, img_dim).cuda()
        dummy_input_seg = torch.randn(1, 6, img_dim, img_dim).cuda()

        t_recon_params, _ = count_parameters(teacher_recon)
        t_seg_params, _ = count_parameters(teacher_seg)
        s_recon_params, _ = count_parameters(student_recon)
        s_seg_params, _ = count_parameters(student_seg_m)

        teacher_info = {
            "total_params": t_recon_params + t_seg_params,
            "recon_params": t_recon_params,
            "seg_params": t_seg_params,
            "model_size_mb": get_model_size_mb(teacher_recon)
            + get_model_size_mb(teacher_seg),
            "gpu_mem_mb": measure_gpu_memory(teacher_recon, dummy_input)
            + measure_gpu_memory(teacher_seg, dummy_input_seg),
        }
        student_info = {
            "total_params": s_recon_params + s_seg_params,
            "recon_params": s_recon_params,
            "seg_params": s_seg_params,
            "model_size_mb": get_model_size_mb(student_recon)
            + get_model_size_mb(student_seg_m),
            "gpu_mem_mb": measure_gpu_memory(student_recon, dummy_input)
            + measure_gpu_memory(student_seg_m, dummy_input_seg),
        }

        print(f"\n  📋 模型結構指標:")
        print(f"  {'指標':<30}{'Teacher (origin)':<20}{'Student':<20}{'壓縮比':<12}")
        print(f"  {'─' * 82}")
        compression_params = (
            teacher_info["total_params"] / student_info["total_params"]
            if student_info["total_params"] > 0
            else 0
        )
        compression_size = (
            teacher_info["model_size_mb"] / student_info["model_size_mb"]
            if student_info["model_size_mb"] > 0
            else 0
        )
        compression_gpu = (
            teacher_info["gpu_mem_mb"] / student_info["gpu_mem_mb"]
            if student_info["gpu_mem_mb"] > 0
            else 0
        )
        print(
            f"  {'總參數量':<28}{teacher_info['total_params']:>14,}{student_info['total_params']:>20,}{compression_params:>12.2f}x"
        )
        print(
            f"  {'  - 重建模型':<28}{teacher_info['recon_params']:>14,}{student_info['recon_params']:>20,}"
        )
        print(
            f"  {'  - 分割模型':<28}{teacher_info['seg_params']:>14,}{student_info['seg_params']:>20,}"
        )
        print(
            f"  {'模型大小 (MB)':<26}{teacher_info['model_size_mb']:>14.2f} MB{student_info['model_size_mb']:>16.2f} MB{compression_size:>12.2f}x"
        )
        print(
            f"  {'GPU 記憶體 (MB)':<25}{teacher_info['gpu_mem_mb']:>14.2f} MB{student_info['gpu_mem_mb']:>16.2f} MB{compression_gpu:>12.2f}x"
        )

        # =====================
        # GPU 暖機
        # =====================
        print(f"\n  🔥 GPU 暖機中...")
        warm_up_model(teacher_recon, dummy_input)
        warm_up_model(teacher_seg, dummy_input_seg)
        warm_up_model(student_recon, dummy_input)
        warm_up_model(student_seg_m, dummy_input_seg)

        # =====================
        # 推論效率測量
        # =====================
        print(f"  ⏱️  測量推論時間中 (重複 {n_repeat} 次)...")

        # 完整 pipeline
        teacher_result = benchmark_inference(
            teacher_recon, teacher_seg, dataloader, n_repeat, "Teacher (origin)"
        )
        student_result = benchmark_inference(
            student_recon, student_seg_m, dataloader, n_repeat, "Student"
        )

        # 各階段拆解
        t_recon_bench = benchmark_recon_only(
            teacher_recon, dataloader, n_repeat, "Teacher Recon"
        )
        t_seg_bench = benchmark_seg_only(
            teacher_recon, teacher_seg, dataloader, n_repeat, "Teacher Seg"
        )
        s_recon_bench = benchmark_recon_only(
            student_recon, dataloader, n_repeat, "Student Recon"
        )
        s_seg_bench = benchmark_seg_only(
            student_recon, student_seg_m, dataloader, n_repeat, "Student Seg"
        )

        # =====================
        # 輸出比較結果
        # =====================
        speedup = (
            teacher_result["avg_time_ms"] / student_result["avg_time_ms"]
            if student_result["avg_time_ms"] > 0
            else float("inf")
        )
        fps_ratio = (
            student_result["fps"] / teacher_result["fps"]
            if teacher_result["fps"] > 0
            else float("inf")
        )

        print(f"\n  📊 推論效率比較結果:")
        print(f"  {'指標':<30}{'Teacher (origin)':<20}{'Student':<20}{'加速比':<12}")
        print(f"  {'─' * 82}")
        print(f"  {'完整 Pipeline:'}")
        print(
            f"  {'  平均推論時間 (ms)':<28}{teacher_result['avg_time_ms']:>14.2f} ms{student_result['avg_time_ms']:>16.2f} ms{speedup:>12.2f}x"
        )
        print(
            f"  {'  標準差 (ms)':<28}{teacher_result['std_time_ms']:>14.2f} ms{student_result['std_time_ms']:>16.2f} ms"
        )
        print(
            f"  {'  FPS':<28}{teacher_result['fps']:>14.1f}{student_result['fps']:>20.1f}{fps_ratio:>12.2f}x"
        )
        print(
            f"  {'  總推論時間 (s)':<28}{teacher_result['total_time_s']:>14.3f} s{student_result['total_time_s']:>17.3f} s"
        )
        print(
            f"  {'  總推論張數':<28}{teacher_result['total_images']:>14}{student_result['total_images']:>20}"
        )

        recon_speedup = (
            t_recon_bench["avg_time_ms"] / s_recon_bench["avg_time_ms"]
            if s_recon_bench["avg_time_ms"] > 0
            else float("inf")
        )
        seg_speedup = (
            t_seg_bench["avg_time_ms"] / s_seg_bench["avg_time_ms"]
            if s_seg_bench["avg_time_ms"] > 0
            else float("inf")
        )

        print(f"\n  {'各階段拆解:'}")
        print(
            f"  {'  重建模型 (ms)':<28}{t_recon_bench['avg_time_ms']:>14.2f} ms{s_recon_bench['avg_time_ms']:>16.2f} ms{recon_speedup:>12.2f}x"
        )
        print(
            f"  {'  分割模型 (ms)':<28}{t_seg_bench['avg_time_ms']:>14.2f} ms{s_seg_bench['avg_time_ms']:>16.2f} ms{seg_speedup:>12.2f}x"
        )

        # =====================
        # 生成比較圖表
        # =====================
        obj_save_dir = os.path.join(save_root, obj_name)
        os.makedirs(obj_save_dir, exist_ok=True)

        plot_comparison(teacher_result, student_result, obj_save_dir, obj_name)
        plot_model_params_comparison(teacher_info, student_info, obj_save_dir, obj_name)
        plot_stage_breakdown(
            t_recon_bench,
            t_seg_bench,
            s_recon_bench,
            s_seg_bench,
            obj_save_dir,
            obj_name,
        )

        # 收集結果
        all_results.append(
            {
                "obj_name": obj_name,
                "teacher_result": teacher_result,
                "student_result": student_result,
                "teacher_info": teacher_info,
                "student_info": student_info,
                "speedup": speedup,
                "compression_params": compression_params,
            }
        )

        # 釋放記憶體
        del teacher_recon, teacher_seg, student_recon, student_seg_m
        torch.cuda.empty_cache()

    # =====================
    # 總結報告
    # =====================
    if len(all_results) > 0:
        print(f"\n{'=' * 80}")
        print("  總結報告")
        print(f"{'=' * 80}")
        print(
            f"  {'物件類別':<20}{'Teacher (ms)':<18}{'Student (ms)':<18}{'加速比':<12}{'參數壓縮比':<12}"
        )
        print(f"  {'─' * 80}")
        for r in all_results:
            print(
                f"  {r['obj_name']:<20}{r['teacher_result']['avg_time_ms']:<18.2f}{r['student_result']['avg_time_ms']:<18.2f}{r['speedup']:<12.2f}x{r['compression_params']:<12.2f}x"
            )

        avg_speedup = np.mean([r["speedup"] for r in all_results])
        avg_compression = np.mean([r["compression_params"] for r in all_results])
        print(f"  {'─' * 80}")
        print(
            f"  {'平均':<20}{'':18}{'':18}{avg_speedup:<12.2f}x{avg_compression:<12.2f}x"
        )

        # 生成總覽圖
        if len(all_results) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                "All Object Categories — Inference Efficiency Overview",
                fontsize=14,
                fontweight="bold",
            )

            obj_names_list = [r["obj_name"] for r in all_results]
            teacher_times = [r["teacher_result"]["avg_time_ms"] for r in all_results]
            student_times = [r["student_result"]["avg_time_ms"] for r in all_results]
            speedups = [r["speedup"] for r in all_results]

            x = np.arange(len(obj_names_list))
            width = 0.35

            ax = axes[0]
            ax.bar(
                x - width / 2,
                teacher_times,
                width,
                label="Teacher",
                color="#e74c3c",
                alpha=0.85,
            )
            ax.bar(
                x + width / 2,
                student_times,
                width,
                label="Student",
                color="#2ecc71",
                alpha=0.85,
            )
            ax.set_ylabel("Avg Inference Time (ms)")
            ax.set_title("Avg Inference Time per Category")
            ax.set_xticks(x)
            ax.set_xticklabels(obj_names_list, rotation=45, ha="right")
            ax.legend()

            ax = axes[1]
            bars = ax.bar(
                obj_names_list, speedups, color="#3498db", alpha=0.85, edgecolor="black"
            )
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Speedup (x)")
            ax.set_title("Student Model Speedup")
            ax.set_xticklabels(obj_names_list, rotation=45, ha="right")
            for bar, val in zip(bars, speedups):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()
            save_path = os.path.join(save_root, "overall_comparison.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"\n  📊 總覽圖已儲存: {save_path}")

    print(f"\n{'=' * 80}")
    print("  🎉 所有比較測試已完成！")
    print(f"  結果儲存於: {os.path.abspath(save_root)}")
    print(f"{'=' * 80}")


# =======================
# Run
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="教師模型 vs 學生模型 推論效率比較")
    parser.add_argument(
        "--obj_id",
        action="store",
        type=int,
        required=True,
        help="物件類別 ID (-1 表示全部)",
    )
    parser.add_argument(
        "--gpu_id",
        action="store",
        type=int,
        default=-2,
        required=False,
        help="GPU ID (-2: auto-select, -1: CPU)",
    )
    parser.add_argument(
        "--mvtec_root", type=str, default="./mvtec", help="MVTec 資料集根目錄"
    )
    parser.add_argument(
        "--n_repeat", type=int, default=3, help="推論重複次數 (取平均，預設3次)"
    )

    args = parser.parse_args()

    if args.gpu_id == -2:
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [
        ["capsule"],
        ["bottle"],
        ["carpet"],
        ["leather"],
        ["pill"],
        ["transistor"],
        ["tile"],
        ["cable"],
        ["zipper"],
        ["toothbrush"],
        ["metal_nut"],
        ["hazelnut"],
        ["screw"],
        ["grid"],
        ["wood"],
    ]

    if int(args.obj_id) == -1:
        picked_classes = [
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
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    if args.gpu_id == -1:
        main(picked_classes, args)
    else:
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
