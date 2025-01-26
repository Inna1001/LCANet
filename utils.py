import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os, sys
import cv2
import random
from prettytable import PrettyTable
import matplotlib.font_manager as fm

# 设置字体为SimSun
font_path = '/usr/share/fonts/truetype/simsun.ttf'  # 替换为宋体的实际路径
my_font = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = ['SimSun']  # 用宋体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

labels_MalImg = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.gen!g', 
          'C2LOP.P', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 
          'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 
          'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A']

labels_BIG2015 = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']

labels_BODMAS = ['benjamin', 'berbew', 'ceeinject', 'dinwod', 'ganelp', 'gepys', 'mira', 'musecador', 'sfone', 'sillyp2p', 'small', 'upatre', 'wabot', 'wacatac']

# 根据输入的数据集名称选择相应的数据集标签
def chooseDataset(dataset):
    if dataset == 'BIG2015':
        labels = labels_BIG2015
    elif dataset == 'MalImg':
        labels = labels_MalImg
    elif dataset == 'BODMAS':
        labels = labels_BODMAS
    else:
        print('Dataset load error, please check!')
        sys.exit()
    return labels

# 数据集类，用于加载恶意软件图像数据集
class MalwareImageDataset(Dataset):
    def __init__(self, dataset_path, dataset):
        self.dataset_path = dataset_path
        self.dataset = dataset
        # 使用 os.listdir 函数获取数据集路径下的所有图像文件名，并将它们保存到 self.images 列表中
        self.images = os.listdir(self.dataset_path)
        # 将图像转换为张量形式
        self.transform = transforms.Compose([
            transforms.ToTensor(),    
        ])
        # 新增：用于记录无法加载的图片路径
        self.failed_images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.dataset_path, image_index)

        try:
            # 尝试读取并转换图片
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or cannot be read at {img_path}")
            image = self.transform(image)
        except FileNotFoundError as e:
            # 图片未找到或无法读取时，记录问题图片路径，并返回一个占位符或默认值
            print(e)
            self.failed_images.append(img_path)
            # 这里可以选择返回一个占位符图像或特殊的标签，例如：
            image = torch.zeros_like(torch.empty(1, 192, 192))  # 假设图像尺寸为192x192 RGB
        
        labels = chooseDataset(self.dataset)

        try:
            label = labels.index(image_index.split('-')[0])
        except ValueError:
            # 如果索引不存在，同样记录并处理
            print(f"Label not found for image {image_index}")
            label = -1  # 或其他默认值

        return image, label
    
    # 可选：提供一个方法来查看或输出所有未能成功加载的图片路径
    def show_failed_images(self):
        if self.failed_images:
            print("Failed to load the following images:")
            for img_path in self.failed_images:
                print(img_path)
        else:
            print("No failed images.")

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels_list: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels_list = labels_list

    def update(self, pred, label):
        for p, t in zip([pred], [label]):
            self.matrix[p, t] += 1

    def summary(self):
        Precision_list = []
        Recall_list = []
        F1_Score_list = []
        Accuracy_list = []

        table = PrettyTable()
        table.field_names = ["", "Accuracy", "Precision", "Recall", "F1 Score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Accuracy = round((TP + TN) / (TP + TN + FN + FP), 5) if TP + TN + FN + FP != 0 else 0.
            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            F1_Score = round((2 * Precision * Recall) / (Precision + Recall), 5) if Precision + Recall != 0 else 0.

            Precision_list.append(Precision)
            Recall_list.append(Recall)
            F1_Score_list.append(F1_Score)
            Accuracy_list.append(Accuracy)

            table.add_row([self.labels_list[i], Accuracy, Precision, Recall, F1_Score])

        macro_Precision = round(np.mean(Precision_list), 5)
        macro_Recall = round(np.mean(Recall_list), 5)
        macro_F1_Score = round(np.mean(F1_Score_list), 5)
        macro_Accuracy = round(np.mean(Accuracy_list), 5)

        table.add_row(['Macro', macro_Accuracy, macro_Precision, macro_Recall, macro_F1_Score])

        print(table)

        return str(macro_Accuracy)

    def plot(self, ConfusionMatrixPath):
        params = {
            'figure.figsize': '9.5, 8',
            'font.size': 14,

            'figure.autolayout': True,
            'savefig.dpi' : 1200,
            'figure.dpi' : 1200,
        }
        plt.rcParams.update(params)

        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Greys)  # 更改为黑白颜色方案

        plt.xticks(range(self.num_classes), self.labels_list, rotation=90, fontproperties=my_font)
        plt.yticks(range(self.num_classes), self.labels_list, fontproperties=my_font)
        plt.colorbar()
        plt.xlabel('真实标签', fontproperties=my_font)
        plt.ylabel('预测标签', fontproperties=my_font)

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black",
                         fontproperties=my_font)
        plt.tight_layout()
        plt.savefig(ConfusionMatrixPath)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num
