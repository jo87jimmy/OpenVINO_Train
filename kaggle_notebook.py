# %%
# ===========================
# 0. GPU 與時間監控功能
# ===========================

import time, threading, torch, subprocess, os

SESSION_LIMIT_HOURS = 20  # Kaggle 預設 GPU 時間限制
start_time = time.time()

def monitor_gpu():
    while True:
        elapsed = time.time() - start_time
        remaining = SESSION_LIMIT_HOURS * 3600 - elapsed
        remaining_h = max(0, remaining // 3600)
        remaining_m = max(0, (remaining % 3600) // 60)

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        try:
            gpu_mem_info = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", shell=True
            ).decode().strip().split("\n")[0]
            mem_used, mem_total = gpu_mem_info.split(", ")
        except Exception:
            mem_used, mem_total = "?", "?"

        print(f"[GPU監控] GPU: {gpu_name} | 記憶體: {mem_used}/{mem_total} MB | "
              f"已運行: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m | "
              f"剩餘: {int(remaining_h)}h {int(remaining_m)}m")
        if remaining <= 1800:
            print("⚠️ 剩餘時間不足30分鐘，建議停止或儲存進度！")
        time.sleep(300)

threading.Thread(target=monitor_gpu, daemon=True).start()

# %%
# ===========================
# 1. 安裝環境
# ===========================

# Kaggle 預裝 PyTorch + CUDA，通常不需要重裝
# 如果遇到版本不匹配，取消下面兩行的註解：
# !pip uninstall -y torch torchvision torchaudio
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

!pip install -q numpy scipy scikit-learn pillow pandas tqdm
!pip install -q pydrive2 google-auth oauth2client
!pip install -q imgaug
!pip install -q opencv-python-headless

# ── 安裝 Anomalib (OpenVINO 團隊維護的異常檢測框架) ──
!pip install -q anomalib

# %%
# ===========================
# 2. 環境變數設定
# ===========================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib.pyplot as plt

# %%
# ===========================
# 3. 下載 INFERENCE_COMPARISON 專案
# ===========================

REPO_DIR = "INFERENCE_COMPARISON"

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/jo87jimmy/INFERENCE_COMPARISON.git
else:
    print(f"✅ '{REPO_DIR}' 已存在，跳過 clone。")

%cd {REPO_DIR}

# %%
# ===========================
# 4. 掛載 MVTec 資料集
# ===========================

CORRECT_DATA_PATH = "/kaggle/input/data-mvtec/mvtec"
LINK_NAME = "mvtec"

if os.path.lexists(LINK_NAME):
    os.remove(LINK_NAME)
    print(f"🗑️ 已移除舊的 '{LINK_NAME}' 連結。")

if os.path.exists(CORRECT_DATA_PATH):
    !ln -s {CORRECT_DATA_PATH} {LINK_NAME}
    print(f"✅ 已成功將 {CORRECT_DATA_PATH} 連結至 ./{LINK_NAME}")
else:
    print(f"❌ 錯誤：找不到 {CORRECT_DATA_PATH}，請確認 Kaggle 資料集是否已正確掛載。")

print("\n--- 檢查 'mvtec' 資料夾內容 ---")
!ls -l mvtec | head -n 10

# %%
# ===========================
# 5. 掛載 Student Model Checkpoints
# ===========================

def create_symlink(src, dst):
    """建立符號連結，若舊的已存在則先刪除"""
    if os.path.lexists(dst):
        os.remove(dst)
        print(f"🗑️ 已移除舊的連結: {dst}")

    if os.path.exists(src):
        os.symlink(src, dst)
        print(f"✅ 已成功將 {src} 連結至 {dst}")
    else:
        print(f"❌ 錯誤：找不到 {src}，請確認 Kaggle 資料集是否已正確掛載。")

# --- Student Model ---
STUDENT_CP_PATH = "/kaggle/input/student-model-checkpoints"
STUDENT_CP_LINK = "student_model_checkpoints"
create_symlink(STUDENT_CP_PATH, STUDENT_CP_LINK)

print("\n--- 檢查 Student Model 檔案 ---")
!ls -lh {STUDENT_CP_LINK}

# %%
# ===========================
# 6. 掛載 DRAEM Checkpoints
# ===========================

DRAEM_CP_PATH = "/kaggle/input/draem-checkpoints/DRAEM_checkpoints"
DRAEM_CP_LINK = "DRAEM_checkpoints"
create_symlink(DRAEM_CP_PATH, DRAEM_CP_LINK)

print("\n--- 檢查 DRAEM Model 檔案 ---")
!ls -lh {DRAEM_CP_LINK}

# %%
# ===========================
# 7. 驗證所有路徑與環境
# ===========================

print("=" * 60)
print("環境驗證")
print("=" * 60)

# 檢查 GPU
print(f"\n🖥️  GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA 版本: {torch.version.cuda}")

# 檢查必要檔案
print(f"\n📁 MVTec 資料集: {'✅ 存在' if os.path.exists('mvtec') else '❌ 不存在'}")
print(f"📁 Student checkpoints: {'✅ 存在' if os.path.exists('student_model_checkpoints') else '❌ 不存在'}")
print(f"📁 DRAEM checkpoints: {'✅ 存在' if os.path.exists('DRAEM_checkpoints') else '❌ 不存在'}")
print(f"📄 main.py: {'✅ 存在' if os.path.exists('main.py') else '❌ 不存在'}")
print(f"📄 train_anomalib_baselines.py: {'✅ 存在' if os.path.exists('train_anomalib_baselines.py') else '❌ 不存在'}")
print(f"📄 benchmark_all.py: {'✅ 存在' if os.path.exists('benchmark_all.py') else '❌ 不存在'}")

# 檢查 anomalib 是否安裝成功
try:
    import anomalib
    print(f"\n📦 anomalib 版本: {anomalib.__version__}")
except ImportError:
    print("\n❌ anomalib 未安裝，請執行 pip install anomalib")

# 列出 MVTec 物件類別
if os.path.exists('mvtec'):
    objs = sorted([d for d in os.listdir('mvtec') if os.path.isdir(os.path.join('mvtec', d))])
    print(f"\n📋 MVTec 物件類別 ({len(objs)} 個): {objs}")

# 列出可用的 checkpoint 檔案
for cp_dir in ['student_model_checkpoints', 'DRAEM_checkpoints']:
    if os.path.exists(cp_dir):
        files = os.listdir(cp_dir)
        print(f"\n📋 {cp_dir} ({len(files)} 個檔案):")
        for f in sorted(files)[:10]:
            print(f"   - {f}")
        if len(files) > 10:
            print(f"   ... 共 {len(files)} 個檔案")

print("\n" + "=" * 60)

# %%
# ========================================
# 8. 訓練 Anomalib Baseline 模型
# ========================================
# 訓練 PatchCore, CFlow, RD4AD, EfficientAD
# 這些模型使用 OpenVINO 團隊的 anomalib 框架
# 資料集: MVTec (與 DRAEM/Student 使用相同資料集)
# ========================================

print("=" * 70)
print("  開始訓練 Anomalib Baseline 模型")
print("  模型: PatchCore, CFlow, RD4AD (Reverse Distillation), EfficientAD")
print("=" * 70)

# ── 訓練全部 4 個模型 × 15 類別 ──
# 預估時間: PatchCore (~5min), CFlow (~30min), RD4AD (~60min), EfficientAD (~40min)
!python train_anomalib_baselines.py \
    --data_root ./mvtec \
    --output_dir ./anomalib_results

# %%
# ========================================
# 8b. (選擇性) 只訓練特定模型或類別
# ========================================

# 只訓練 PatchCore 和 EfficientAD，只在 bottle 和 carpet 上:
# !python train_anomalib_baselines.py \
#     --data_root ./mvtec \
#     --output_dir ./anomalib_results \
#     --models PatchCore EfficientAD \
#     --categories bottle carpet

# %%
# ========================================
# 9. 原始推論比較 (Teacher vs Student only)
# ========================================

# 測試單一類別 (例如 bottle, obj_id=1)
!python main.py --obj_id 1 --mvtec_root ./mvtec --n_repeat 3

# %%
# ========================================
# 10. 統一 Benchmark：所有模型綜合比較
# ========================================
# 比較 6 個模型:
#   1. PatchCore   (Anomalib)
#   2. CFlow       (Anomalib)
#   3. RD4AD       (Anomalib)
#   4. EfficientAD (Anomalib)
#   5. DRAEM       (Teacher — 自訂模型)
#   6. Ours        (Student — 壓縮模型)
#
# 指標:
#   Image-AUROC, Pixel-AUROC, Image-AP, Pixel-AP
#   Inference Time (ms), FPS
# ========================================

print("=" * 70)
print("  統一 Benchmark：6 個模型 × 15 類別 × 6 指標")
print("=" * 70)

!python benchmark_all.py \
    --data_root ./mvtec \
    --anomalib_dir ./anomalib_results \
    --draem_dir ./DRAEM_checkpoints \
    --student_dir ./student_model_checkpoints \
    --save_dir ./benchmark_results \
    --n_repeat 3

# %%
# ========================================
# 10b. (選擇性) 只 benchmark 特定類別
# ========================================

# !python benchmark_all.py \
#     --data_root ./mvtec \
#     --anomalib_dir ./anomalib_results \
#     --draem_dir ./DRAEM_checkpoints \
#     --student_dir ./student_model_checkpoints \
#     --save_dir ./benchmark_results \
#     --categories bottle carpet pill \
#     --n_repeat 3

# %%
# ========================================
# 11. 顯示原始比較結果
# ========================================

import glob
from IPython.display import display, Image as IPImage

results_dir = "./comparison_results"

if os.path.exists(results_dir):
    png_files = sorted(glob.glob(f"{results_dir}/**/*.png", recursive=True))
    print(f"找到 {len(png_files)} 張結果圖表：\n")
    for png_file in png_files:
        print(f"📊 {png_file}")
        display(IPImage(filename=png_file))
        print()
else:
    print("❌ 尚未產生原始比較結果。")

# %%
# ========================================
# 12. 顯示統一 Benchmark 結果
# ========================================

benchmark_dir = "./benchmark_results"

if os.path.exists(benchmark_dir):
    # 顯示摘要表格
    summary_img = os.path.join(benchmark_dir, "summary_table.png")
    if os.path.exists(summary_img):
        print("📊 所有模型平均指標摘要:")
        display(IPImage(filename=summary_img))
        print()

    # 顯示各指標比較圖
    metric_charts = [
        "comparison_image_auroc.png",
        "comparison_pixel_auroc.png",
        "comparison_image_ap.png",
        "comparison_pixel_ap.png",
        "comparison_inference_time.png",
        "comparison_fps.png",
    ]
    for chart_name in metric_charts:
        chart_path = os.path.join(benchmark_dir, chart_name)
        if os.path.exists(chart_path):
            print(f"📊 {chart_name}:")
            display(IPImage(filename=chart_path))
            print()

    # 顯示 CSV 結果
    csv_path = os.path.join(benchmark_dir, "benchmark_results.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        print("\n📄 完整 Benchmark 數據:")
        df = pd.read_csv(csv_path)
        display(df)
else:
    print("❌ 尚未產生 benchmark 結果，請先執行步驟 8 和 10。")

# %%
# ========================================
# 13. 下載所有結果
# ========================================

# 打包所有結果
for results in ["comparison_results", "benchmark_results", "anomalib_results/checkpoints"]:
    if os.path.exists(results):
        zip_name = results.replace("/", "_")
        !zip -r /kaggle/working/{zip_name}.zip {results}
        print(f"✅ {results} 已打包至 /kaggle/working/{zip_name}.zip")

print("\n📥 可從 Kaggle Output 頁面下載所有結果。")
