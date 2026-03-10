import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from models.RRDBNet_arch import RRDBNet

# ✅ 1. 设置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'models/RRDB_ESRGAN_x4.pth'  # 原始 ESRGAN 模型路径
train_dir = 'datasets/DIV2K/DIV2K_train_HR'     # HR 图像目录
save_path = 'models/RRDB_CBAM_finetuned.pth'

# ✅ 2. 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ✅ 3. 加载模型（带 CBAM）
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)

# 加载预训练参数（跳过 CBAM 部分）
pretrained_dict = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_dict, strict=False)

# ✅ 4. 损失函数 & 优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 5. 微调训练
print("🚀 开始训练...")
for epoch in range(5):  # 微调5轮即可
    model.train()
    for i, (imgs, _) in enumerate(train_loader):
        hr = imgs.to(device)
        lr = nn.functional.interpolate(hr, scale_factor=1/4, mode='bicubic', align_corners=False)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Loss {loss.item():.4f}")

# ✅ 6. 保存模型
torch.save(model.state_dict(), save_path)
print(f"✅ 训练完成，模型已保存：{save_path}")
