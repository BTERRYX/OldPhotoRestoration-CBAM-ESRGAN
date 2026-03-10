import os

img_folder = "datasets/DIV2K/DIV2K_valid_HR"
all_files = os.listdir(img_folder)
print("文件夹中共有文件：", len(all_files))
for file in all_files:
    print(file)

valid_imgs = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print("有效图片文件数量：", len(valid_imgs))
