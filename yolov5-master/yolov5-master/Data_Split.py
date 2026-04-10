import os
import random
import shutil

# 数据集路径
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 划分比例
train_ratio = 0.8  # 80%训练，20%验证

# 创建目标文件夹
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# 获取所有图片文件名（不带扩展名）
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
image_basenames = [os.path.splitext(f)[0] for f in image_files]

# 随机划分
random.shuffle(image_basenames)
num_train = int(len(image_basenames) * train_ratio)
train_basenames = image_basenames[:num_train]
val_basenames = image_basenames[num_train:]

def move_files(basenames, split):
    for name in basenames:
        # 移动图片
        for ext in ['.jpg', '.png']:
            img_src = os.path.join(images_dir, name + ext)
            img_dst = os.path.join(images_dir, split, name + ext)
            if os.path.exists(img_src):
                shutil.move(img_src, img_dst)
        # 移动标签
        label_src = os.path.join(labels_dir, name + '.txt')
        label_dst = os.path.join(labels_dir, split, name + '.txt')
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)

move_files(train_basenames, 'train')
move_files(val_basenames, 'val')

print(f'划分完成：训练集 {len(train_basenames)}，验证集 {len(val_basenames)}')