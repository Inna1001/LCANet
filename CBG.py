import os
import cv2
import numpy as np


def data_augmentation(gray_image):
    # 对灰度图像直接应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # 应用双三次插值法
    height, width = enhanced_image.shape
    enhanced_image = cv2.resize(enhanced_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # 结合高斯模糊
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    return enhanced_image


if __name__ == "__main__":
    dataset_dir = './datasets/MalImg'
    # dataset_dir = './datasets/BIG2015'
    # dataset_dir = './datasets/BODMAS'
    dataset_CBG_dir = './datasets/MalImg/MalImg-CBG'

    # 检查dataset_dir是否存在
    if not os.path.exists(dataset_dir):
        print(f"Error: 数据集目录 {dataset_dir} 不存在!")
        exit(1)

    # 检查dataset_CBG_dir是否存在，如果不存在则创建
    if not os.path.exists(dataset_CBG_dir):
        os.makedirs(dataset_CBG_dir)

    # print(f"开始处理数据集目录: {dataset_dir}")

    for img in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img)

        # 确保img_path是一个文件而不是目录
        if os.path.isdir(img_path):
            print(f"跳过子目录: {img_path}")
            continue

        # 跳过已处理的图像
        if '_processed' in img:
            print(f"跳过已处理的图像: {img_path}")
            continue

        # 读取灰度图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 打印原始图像的尺寸
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue

        print(f"处理图像: {img_path}, 尺寸: {image.shape}")

        try:
            CBG_image = data_augmentation(image)  # 自定义函数实现数据增强

            # 输出增强图像的尺寸
            print(f"增强后的图像: {os.path.join(dataset_CBG_dir, img)}, 尺寸: {CBG_image.shape}")

            # 保存灰度图像
            output_path = os.path.join(dataset_CBG_dir, img.replace('.png', '_processed.png'))
            cv2.imwrite(output_path, aug_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        except Exception as e:
            print(f"处理文件 {img_path} 时发生错误: {str(e)}")

    print("处理完成")