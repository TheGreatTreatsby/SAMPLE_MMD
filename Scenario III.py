import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
# 自定义噪声添加函数
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')  # 打开图像
        if self.transform:
            img = self.transform(img)

        return img, label

class ImagePathDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.samples = list(image_folder.imgs)  # 获取图像路径和标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')  # 打开图像
        if self.transform:
            img = self.transform(img)


        return img, label, img_path  # 返回图像 tensor、标签和路径

class AddNoise(object):
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor

    def __call__(self, img):
        noise = torch.randn_like(img) * self.noise_factor
        noisy_img = torch.clamp(img + noise, 0, 1)
        return noisy_img

def create_combined_dataloader(dataset1, num_samples_to_extract, batch_size=128, num_workers=0):
    """
    从 dataset1 中随机抽取指定数量的样本，创建 dataset2，并将 dataset1 和 dataset2 组合成一个新的数据集，
    最后返回对应的 DataLoader。

    :param dataset1: 原始数据集（torch.utils.data.Dataset）
    :param num_samples_to_extract: 从 dataset1 中抽取的样本数量
    :param batch_size: DataLoader 的批量大小
    :param num_workers: DataLoader 的工作线程数
    :return: 组合后的 DataLoader
    """
    dataset1 = random_split(dataset1, [len(dataset1)])[0]
    indices = list(range(len(dataset1)))


    # 计算 dataset1 的完整 batch 数量和剩余样本数量
    num_full_batches = len(dataset1) // batch_size
    remaining_samples = len(dataset1) % batch_size
    supply_samples=batch_size-remaining_samples


    # 划分 dataset1 的完整 batch 和剩余样本
    full_batch_indices = indices[:num_full_batches * batch_size]

    # 从完整 batch 中随机抽取样本，用于填充 dataset2 的前部分
    dataset2_indices_part1 = random.sample(full_batch_indices, supply_samples)

    # 从剩余样本中随机抽取样本，用于填充 dataset2 的后部分
    num_samples_from_remaining = num_samples_to_extract - (num_full_batches+1) * batch_size
    if num_samples_from_remaining>len(dataset1):
        dataset2_indices_part2 = random.sample(indices, num_full_batches*batch_size)
        dataset2_indices_part3 = random.sample(indices, num_samples_to_extract-(2*num_full_batches+1) * batch_size)
        dataset2_indices = dataset2_indices_part1 + dataset2_indices_part2+dataset2_indices_part3

    else:
        dataset2_indices_part2 = random.sample(indices, num_samples_from_remaining)
        # 合并 dataset2 的索引
        dataset2_indices = dataset2_indices_part1 + dataset2_indices_part2

    # 创建 dataset2
    dataset2 = Subset(dataset1, dataset2_indices)
    # 组合 dataset1 和 dataset2
    combined_dataset = ConcatDataset([dataset1, dataset2])
    # 创建 DataLoader，不洗牌
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 打印数据集大小
    # print(f"Dataset1 size: {len(dataset1)}")
    # print(f"Dataset2 size: {len(dataset2)}")
    # print(f"Combined dataset size: {len(combined_dataset)}")
    return dataloader

# 定义带适应层的ResNet-18
class AdaptedResNet(nn.Module):
    def __init__(self, num_classes=10, adaptation_dim=128, dropout_rate=0.0):
        super(AdaptedResNet, self).__init__()
        # 加载预训练ResNet-18
        base_model = models.resnet18(pretrained=False)
        #  dropout 概率为0
        if dropout_rate > 0:
            for layer in [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]:
                layer.add_module('dropout', nn.Dropout(p=dropout_rate))
        # 提取特征层（移除最后的全连接层）
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 添加适应层
        self.adaptation_layer = nn.Linear(base_model.fc.in_features, adaptation_dim)
        # 分类层
        self.fc = nn.Linear(adaptation_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_adapt = self.adaptation_layer(x)  # 适应层输出
        x_class = self.fc(x_adapt)  # 分类输出
        return x_class, x_adapt


def gaussian_kernel(x, y, sigma=1.0):
    """
    计算高斯核函数
    :param x: 输入张量，形状为 (batch_size, feature_dim)
    :param y: 输入张量，形状为 (batch_size, feature_dim)
    :param sigma: 高斯核的带宽参数
    :return: 高斯核矩阵，形状为 (batch_size, batch_size)
    """
    x = x.view(x.size(0), -1)  # 展平为 (batch_size, feature_dim)
    y = y.view(y.size(0), -1)  # 展平为 (batch_size, feature_dim)
    # 计算 pairwise 的 L2 距离
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)  # (batch_size, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)  # (1, batch_size)
    dist = x_norm + y_norm - 2 * torch.mm(x, y.t())  # (batch_size, batch_size)
    # 计算高斯核
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel


def MMD_loss_gaussian(X, Y, sigma=1.0):

    """
    计算两个矩阵 X 和 Y 之间的最大均值差异（MMD）。

    参数:
        X (torch.Tensor): 形状为 (batch_size, dim) 的矩阵。
        Y (torch.Tensor): 形状为 (batch_size, dim) 的矩阵。
        sigma (float): 高斯核的带宽参数。

    返回:
        torch.Tensor: MMD 的值。
    """
    batch_size_X = X.size(0)
    batch_size_Y = Y.size(0)

    # 计算核矩阵
    K_XX = gaussian_kernel(X, X, sigma)
    K_YY = gaussian_kernel(Y, Y, sigma)
    K_XY = gaussian_kernel(X, Y, sigma)
    source_source=  (K_XX.sum() - K_XX.trace()) / (batch_size_X * (batch_size_X - 1))
    target_target=(K_YY.sum() - K_YY.trace()) / (batch_size_Y * (batch_size_Y - 1))
    source_target=K_XY.mean()
    # 计算 MMD 的无偏估计
    mmd = source_source+target_target-2*source_target
    return mmd, source_source, target_target, source_target

# MMD损失函数（线性核）
def MMD_loss(source, target):
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)
    # 计算源域样本对的内积矩阵
    source_source = torch.mm(source, source.t())# (batch_size, batch_size)
    source_source=source_source.mean()
    # 计算目标域样本对的内积矩阵
    target_target = torch.mm(target, target.t()) # (batch_size, batch_size)
    target_target = target_target.mean()
    # 计算源域和目标域样本对的内积矩阵
    source_target = torch.mm(source, target.t()).mean()   # (batch_size, batch_size)
    mmd = torch.exp(torch.abs(source_source+target_target-2*source_target))-1
    # mmd = torch.exp(source_source_non_diagonal + target_target_non_diagonal - 2 * source_target)
    # mmd = torch.abs(source_source + target_target - 2 * source_target)
    # mmd =torch.log(1 + source_source + target_target - 2 * source_target)
    return mmd, source_source, target_target, source_target

def MMD_loss_unbiased(source, target):
    batch_size_X = source.size(0)
    batch_size_Y = target.size(0)

    source = source.view(source.size(0), -1)  # 展平为 (batch_size, feature_dim)
    target = target.view(target.size(0), -1)  # 展平为 (batch_size, feature_dim)

    # 计算源域样本对的内积矩阵
    source_source = torch.mm(source, source.t())  # (batch_size, batch_size)
    # 计算目标域样本对的内积矩阵
    target_target = torch.mm(target, target.t())  # (batch_size, batch_size)
    # 计算源域和目标域样本对的内积矩阵
    source_target = torch.mm(source, target.t())  # (batch_size, batch_size)
    # 提取对角线元素并计算均值
    source_source_non_diagonal = (source_source.sum() - source_source.trace()) / (batch_size_X * (batch_size_X - 1))
    target_target_non_diagonal = (target_target.sum() - target_target.trace()) / (batch_size_Y * (batch_size_Y - 1))
    source_target_mean = source_target.mean()  # 源域和目标域的均值

    # 计算 MMD 损失
    mmd = torch.abs(source_source_non_diagonal + target_target_non_diagonal - 2 * source_target_mean)
    return mmd,source_source_non_diagonal ,target_target_non_diagonal ,source_target_mean
# 计算准确率
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return 100 * correct / labels.size(0)


def custom_lr_scheduler(optimizer, epoch,mode=2):#两种学习策略均可使用，模式二较稳定
    if mode==1:
        epoch_in_cycle = epoch % 80  # 周期内的epoch（0-79）

        if epoch_in_cycle < 40:
            lr = 0.001
        elif epoch_in_cycle < 60:
            lr = 0.001 * 0.5  # 第40epoch衰减
        else:
            lr = 0.001 * 0.25  # 第60epoch衰减
    if mode==2:
        if epoch < 80:
            lr = 0.001
        elif epoch < 160:
            lr = 0.001 * 0.5  # 第80epoch衰减
        else:
            lr = 0.001 * 0.25  # 第160epoch衰减

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(lamada,adaptation_dim,denoise_step=[85,95],kmeans=False,stage=[80,160],scheduler=True,mode=1):
    freeze_support()

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # AddNoise(noise_factor=0.4),  # 噪声因子
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 场景3：加载源域和目标域数据
    source_dataset = datasets.ImageFolder(root='D:\dataset\SAR\SAMPLE_dataset_public-master\png_images\qpm\\synth', transform=train_transform)
    target_train_dataset = ImagePathDataset(datasets.ImageFolder(root='D:\dataset\SAR\SAMPLE_dataset_public-master\png_images\qpm\\real_elev_14_16'),transform=test_transform)
    target_test_dataset = datasets.ImageFolder(root='D:\dataset\SAR\SAMPLE_dataset_public-master\png_images\qpm\\real_elev_17', transform=test_transform)


    # 创建 DataLoader
    source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
    target_train_loader_raw = DataLoader(target_train_dataset, batch_size=128, shuffle=True, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=128, shuffle=True, num_workers=0)
    print(len(source_dataset))
    print(len(target_train_dataset))
    print(len(target_test_dataset))

    # 初始化模型、损失函数和优化器
    model = AdaptedResNet(num_classes=10,adaptation_dim=adaptation_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lambda_mmd = lamada  # 调整 MMD 损失权重

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建以时间日期命名的文件夹
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_time = os.path.join('result', current_time)
    os.makedirs(current_time, exist_ok=True)

    # 训练循环
    num_epochs = 240
    stage_id=0

    train_losses = []  # 总损失
    cls_losses = []    # 分类损失
    mmd_losses = []    # MMD 损失
    source_source_values = []  # source_source 值
    target_target_values = []  # target_target 值
    source_target_values = []  # source_target 值
    train_accuracies = []  # 准确率

    for epoch in range(num_epochs):
        # 更新学习率
        if scheduler:
            optimizer = custom_lr_scheduler(optimizer, epoch,mode=mode)

        # 打印当前学习率（调试用）
        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        running_source_source = 0.0
        running_target_target = 0.0
        running_source_target = 0.0
        running_accuracy = 0.0
        target_train_loader=create_combined_dataloader(target_train_dataset, 1345)
        batch_num=0
        total_train=0
        for (source_inputs, source_labels), (target_inputs, target_labels,_) in zip(source_loader, target_train_loader):
            total_train += source_labels.size(0)
            batch_num+=1
            source_inputs = source_inputs.to(device)
            source_labels = source_labels.to(device)
            target_inputs = target_inputs.to(device)
            source_outputs, source_adapt = model(source_inputs)
            _, target_adapt = model(target_inputs)
            cls_loss = criterion(source_outputs, source_labels)
            mmd_loss, source_source, target_target, source_target = MMD_loss(source_adapt, target_adapt)
            # mmd_loss, source_source, target_target, source_target = MMD_loss_unbiased(source_adapt, target_adapt)
            # mmd_loss, source_source, target_target, source_target  = MMD_loss_gaussian(source_adapt, target_adapt)

            total_loss = cls_loss + lambda_mmd * mmd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_mmd_loss += mmd_loss.item()
            running_source_source += source_source.item()
            running_target_target += target_target.item()
            running_source_target += source_target.item()
            running_accuracy += calculate_accuracy(source_outputs, source_labels)


        epoch_loss = running_loss / batch_num
        epoch_cls_loss = running_cls_loss / batch_num
        epoch_mmd_loss = running_mmd_loss / batch_num
        epoch_source_source = running_source_source /batch_num
        epoch_target_target = running_target_target / batch_num
        epoch_source_target = running_source_target / batch_num
        epoch_accuracy = running_accuracy / batch_num

        train_losses.append(epoch_loss)
        cls_losses.append(epoch_cls_loss)
        mmd_losses.append(epoch_mmd_loss)
        source_source_values.append(epoch_source_source)
        target_target_values.append(epoch_target_target)
        source_target_values.append(epoch_source_target)
        train_accuracies.append(epoch_accuracy)
        if (epoch+1)%10==0:

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'LR: {current_lr:.6f},'
                  f'Total Loss: {epoch_loss:.4f}, '
                  f'Cls Loss: {epoch_cls_loss:.4f}, '
                  f'MMD Loss: {epoch_mmd_loss:.4f}, '
                  f'source_source: {epoch_source_source:.4f}, '
                  f'target_target: {epoch_target_target:.4f}, '
                  f'source_target: {epoch_source_target:.4f}, ',
                  f'Accuracy: {epoch_accuracy:.2f}%')

        if (epoch + 1) in stage :
            denoise_threshold=denoise_step[stage_id]
            stage_id+=1
            lambda_mmd*=1
            print('denoise_threshold',denoise_threshold)
            # print('lambda_mmd', lambda_mmd)
            source_features = []
            source_labels_list = []
            target_features = []
            target_labels_list = []
            target_pseudo_labels=[]
            target_image_paths = []

            # 测试目标域测试集性能
            model.eval()
            with torch.no_grad():
                for inputs, labels, paths in target_train_loader_raw:#从目标域中抽取特征
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, features = model(inputs)
                    target_features.append(features.cpu())
                    _, predicted = torch.max(outputs.data, 1)
                    target_pseudo_labels.append(predicted.cpu())
                    target_image_paths.extend(paths)  # 保存图像路径
                    target_labels_list.append(labels.cpu())

            # 提取特征并进行t-SNE可视化
            model.eval()
            with torch.no_grad():
                for inputs, labels in source_loader:#从源域中抽取特征
                    inputs = inputs.to(device)
                    _, features = model(inputs)
                    source_features.append(features.cpu())
                    source_labels_list.append(labels)

            source_features = torch.cat(source_features, dim=0)
            target_features = torch.cat(target_features, dim=0)
            source_labels = torch.cat(source_labels_list, dim=0)
            target_labels = torch.cat(target_labels_list, dim=0)
            target_pseudo_labels = torch.cat(target_pseudo_labels, dim=0)


            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            target_features_2d = tsne.fit_transform(target_features.numpy())
            # 对每类进行 K-Means 聚类
            if kmeans:
                kmeans = KMeans(n_clusters=1)  # 每类只有一个聚类中心,等价于质心计算
            # 计算每类的聚类中心
            class_centers = []
            for i in range(10):
                class_mask = (target_pseudo_labels == i)
                if class_mask.sum() > 0:  # 确保该类有样本
                    if kmeans:
                        kmeans.fit(target_features_2d[class_mask])
                        class_centers.append(kmeans.cluster_centers_[0])
                    else:
                        class_centers.append(target_features_2d[class_mask].mean(axis=0))
                else:
                    class_centers.append(np.zeros(2))  # 如果没有样本，用零向量代替

            # 计算每个点到其聚类中心的距离
            distances = []
            for i in range(len(target_features_2d)):
                label = target_pseudo_labels[i]
                center = class_centers[label]
                distance = np.linalg.norm(target_features_2d[i] - center)
                distances.append(distance)

            # 根据距离筛选出前N%的点
            threshold = np.percentile(distances, denoise_threshold)
            selected_mask = np.array(distances) <= threshold

            # 筛选出可信的伪标签和数据

            selected_pseudo_labels = target_pseudo_labels[selected_mask]
            selected_target_labels = target_labels[selected_mask]
            selected_image_paths = [target_image_paths[i] for i in range(len(target_image_paths)) if selected_mask[i]]
            print(len(selected_image_paths))
            print(len(selected_pseudo_labels))
            print(len(selected_target_labels))
            print('伪标签正确率',np.sum(np.array(selected_pseudo_labels) == np.array(selected_target_labels))/len(selected_pseudo_labels))

            # 构建新的数据集

            new_dataset = CustomDataset(selected_image_paths, selected_pseudo_labels.tolist(), transform=train_transform)
            combined_dataset = ConcatDataset([source_dataset, new_dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=0)

            # 替换原来的源域 DataLoader
            source_loader = combined_loader

            # 合并特征和标签
            all_features = torch.cat([source_features, target_features], dim=0)
            all_labels = torch.cat([source_labels.cpu(), target_labels.cpu()], dim=0)
            domain_labels = torch.cat([torch.zeros(len(source_features)), torch.ones(len(target_features))], dim=0)

            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            all_features_2d = tsne.fit_transform(all_features.numpy())

            # 定义颜色和形状
            colors = plt.cm.tab10.colors  # 10种颜色
            source_marker = 'o'  # 合成图像用圆形
            target_marker = '^'  # 真实图像用三角形

            # t-sne可视化
            plt.figure(figsize=(12, 8))
            for i in range(10):  # 遍历10类
                # 绘制合成图像（源域）
                source_mask = (all_labels == i) & (domain_labels == 0)
                plt.scatter(all_features_2d[source_mask, 0], all_features_2d[source_mask, 1],
                            marker=source_marker, color=colors[i], edgecolor='black', linewidths=0.5,
                            label=f'Class {i} (Synthetic)' if i == 0 else "")

                # 绘制真实图像（目标域）
                target_mask = (all_labels == i) & (domain_labels == 1)
                plt.scatter(all_features_2d[target_mask, 0], all_features_2d[target_mask, 1],
                            marker=target_marker, color=colors[i], edgecolor='black', linewidths=0.5,
                            label=f'Class {i} (Real)' if i == 0 else "")

            # 添加图例
            source_legend = mlines.Line2D([], [], color='black', marker=source_marker, linestyle='None',
                                          markersize=10, markerfacecolor='white', markeredgecolor='black',
                                          label='Synthetic (Source)')
            target_legend = mlines.Line2D([], [], color='black', marker=target_marker, linestyle='None',
                                          markersize=10, markerfacecolor='white', markeredgecolor='black',
                                          label='Measure (Target)')
            plt.legend(handles=[source_legend, target_legend])

            plt.title(f't-SNE Visualization at Epoch {epoch + 1}')

            # 保存高清图片
            plt.savefig(os.path.join(current_time, f'{epoch + 1}_epoch_t-SNE.png'), dpi=300, bbox_inches='tight')
            plt.close()

    if epoch+1==240:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in target_test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        final_accuracy = 100 * correct / total
        print(f'Accuracy on {epoch + 1} epoch target test dataset: {final_accuracy}%')

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Total Loss')
    # plt.plot(cls_losses, label='Classification Loss')
    plt.plot(mmd_losses, label='MMD Loss')
    plt.plot(source_source_values, label='Source Source')
    plt.plot(target_target_values, label='Target Target')
    plt.plot(source_target_values, label='Source Target')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and MMD Components')
    plt.legend()

    # 绘制准确度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(current_time, 'training_curves.png'))
    # plt.show()

    # 保存准确度图
    plt.figure()
    plt.bar(['Accuracy'], [final_accuracy])
    plt.ylabel('Accuracy')
    plt.title('Final Accuracy on Target Test Dataset')
    plt.savefig(os.path.join(current_time, f'{final_accuracy:.2f}_accuracy.png'))
    # plt.show()

    return final_accuracy


if __name__ == '__main__':
    for lamada in [0.75]:
        for adaptation_dim in [256]:
            for mode in [2, 1]:
                for denoise_step, stage  in zip([[80, 90, 100],[90, 100], [75, 95],  ], [ [80, 160, 220],[80, 160], [80, 160], ]):
                    result = []
                    file_disk = r'xxx'
                    os.makedirs(file_disk, exist_ok=True)
                    filename = os.path.join(file_disk,f"Scenario3_raw_lr_lamada_{lamada}_adaptation_dim_{adaptation_dim}_denoise_step _{denoise_step}_stage {stage}_mode {mode}acc.txt")
                    print(f"lamada {lamada}, adaptation_dim {adaptation_dim}, denoise_step {denoise_step},stage {stage},start!")

                    for i in range(20):
                        current = train(lamada, adaptation_dim, denoise_step=denoise_step, kmeans=False, stage=stage, mode=mode)
                        result.append(current)
                        print(result)
                        stats = f"min: {np.min(result)}, max: {np.max(result)}, mean: {np.mean(result)}, std: {np.std(result)}"
                        print(stats)
                        # Save current results and stats to file
                        with open(filename, 'w') as f:
                            f.write(f"All results: {result}\n")
                            f.write(f"Statistics: {stats}\n")
                            f.write(f"Final mean accuracy: {np.mean(result)}\n")

                    final_output = f"lamada {lamada}, adaptation_dim {adaptation_dim}, denoise_step {denoise_step},stage {stage}, acc={np.mean(result)}"
                    print(final_output)
                    # Save final results to file
                    with open(filename, 'a') as f:  # 'a' mode to append if needed
                        f.write(f"\nFinal: {final_output}\n")
