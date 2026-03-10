import os
import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.transforms.functional import to_tensor
from PIL import Image

# 根据模型选择结构
from RRDBNet_arch_NOCBAM import RRDBNet_NOCBAM  # 原始模型（无 CBAM）
from models.RRDBNet_arch import RRDBNet  # 你自定义的带 CBAM 的模型类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(pth_path, use_cbam=False):
    print(f"📦 Loading model from: {pth_path} | use_cbam={use_cbam}")
    if use_cbam:
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
    else:
        model = RRDBNet_NOCBAM(in_nc=3, out_nc=3, nf=64, nb=23)
    model.load_state_dict(torch.load(pth_path), strict=True)
    model.eval().to(device)
    return model

def read_image_tensor(path):
    try:
        img = Image.open(path).convert("RGB")
        return to_tensor(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 无法读取图像 {path}: {e}")
        return None

def evaluate_model(model, lr_folder, hr_folder):
    psnr_list, ssim_list = [], []
    image_names = sorted(os.listdir(lr_folder))
    total = 0

    for img_name in image_names:
        lr_path = os.path.join(lr_folder, img_name)
        hr_path = os.path.join(hr_folder, img_name)
        if not os.path.exists(hr_path):
            print(f"⚠️ 缺失 HR 图像: {hr_path}")
            continue

        lr_img = read_image_tensor(lr_path)
        hr_img = read_image_tensor(hr_path)

        if lr_img is None or hr_img is None:
            continue

        total += 1
        with torch.no_grad():
            sr_img = model(lr_img).clamp(0, 1)

        sr_np = sr_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_np = hr_img.squeeze().cpu().numpy().transpose(1, 2, 0)

        psnr_val = compare_psnr(hr_np, sr_np, data_range=1.0)
        ssim_val = compare_ssim(hr_np, sr_np, channel_axis=2, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    if total == 0:
        print("🚫 错误：没有有效的图像对可评估。请检查图像路径和格式。")
        return float('nan'), float('nan')

    return np.mean(psnr_list), np.mean(ssim_list)

# 路径设置
baseline_path = "models/RRDB_ESRGAN_x4.pth"
cbam_path = "models/RRDB_CBAM_finetuned.pth"
lr_folder = "datasets/DIV2K/DIV2K_valid_LR_bicubic/X4"
hr_folder = "datasets/DIV2K/DIV2K_valid_HR"

# 评估 baseline 模型
print("📦 Evaluating Baseline Model...")
model_baseline = load_model(baseline_path, use_cbam=False)
psnr_b, ssim_b = evaluate_model(model_baseline, lr_folder, hr_folder)

# 评估 CBAM 模型
print("\n🔬 Evaluating CBAM Model...")
model_cbam = load_model(cbam_path, use_cbam=True)
psnr_c, ssim_c = evaluate_model(model_cbam, lr_folder, hr_folder)

# 打印结果
print("\n📊 Evaluation Summary:")
print(f"Baseline → PSNR: {psnr_b:.2f}, SSIM: {ssim_b:.4f}")
print(f"CBAM     → PSNR: {psnr_c:.2f}, SSIM: {ssim_c:.4f}")
