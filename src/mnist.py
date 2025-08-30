"""
案例:Pytorch搭建全连接网络，实现MNIST手写识别
1使用DataLoader加载数据集
2定义网络模型，选择损失函数和优化器
3在训练集上训练模型
4使用测试集评估模型
5识别指定图片
"""
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# 定义超参数
epoches = 3  # 总的训练迭代次数
batch_size = 128  # 每个batch加载多少个样本
learning_rate = 0.001  # 学习率
in_dim = 28 * 28  # 样本数据维度 （图片大小28*28）
out_dim = 10  # 输出维度（分类数） 手写数字共10个分类
n_hidden_1 = 500  # 第1个全年连接层的神经元个数
n_hidden_2 = 100  # 第2个全年连接层的神经元个数

# 获取数据集，使用DataLoader加载数据集
train_dataset = datasets.MNIST(root='../static/mnist', train=True, transform=transforms.ToTensor(),
                               download=True)  # 下载MNIST数据集，并转换为张量
test_dataset = datasets.MNIST(root='../static/mnist', train=False, transform=transforms.ToTensor(),
                              download=True)  # 下载MNIST数据集，并转换为张量

print(len(train_dataset), len(test_dataset))
