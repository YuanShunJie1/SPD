import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', default="./checkpoint_spd/images_w=0_poison=50_alpha=50.0_lr=0.01_dev=5.0_backdoor_visualize_compute/", type=str, help='path to dataset')

args = parser.parse_args()



def calculate_psnr(img1, img2):
    """计算两张图像的 PSNR 值"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # 防止 log(0)
        return float('inf')
    max_pixel = 255.0  # 假设图像是8位的
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_metrics_for_directory(dir_path):
    """计算指定目录下所有图像对的 PSNR 和 SSIM"""
    psnr_values = []
    ssim_values = []
    
    for file_name in os.listdir(dir_path):
        if file_name.endswith('_clean.png'):  # 匹配原图文件
            clean_path = os.path.join(dir_path, file_name)
            poison_path = os.path.join(dir_path, file_name.replace('_clean.png', '_poison.png'))
            
            if not os.path.exists(poison_path):
                print(f"Missing pair for: {file_name}")
                continue
            
            # 读取图像（灰度模式或保持一致的通道格式）
            img1 = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(poison_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f"Error reading: {file_name}")
                continue
            
            # 确保尺寸一致
            if img1.shape != img2.shape:
                print(f"Shape mismatch: {file_name}")
                continue
            
            # 计算 PSNR 和 SSIM
            psnr = calculate_psnr(img1, img2)
            ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())
            
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)
    
    # 计算均值
    psnr_mean = np.mean(psnr_values) if psnr_values else None
    ssim_mean = np.mean(ssim_values) if ssim_values else None
    
    print('psnr:',psnr_values)
    print('ssim:',ssim_values)
    print(len(psnr_values))
        
    return psnr_mean, ssim_mean

# 使用示例
directory_path = args.path.strip()  # 替换为你的目录路径
psnr_mean, ssim_mean = calculate_metrics_for_directory(directory_path)

print(f"Average PSNR: {psnr_mean}")
print(f"Average SSIM: {ssim_mean}")
