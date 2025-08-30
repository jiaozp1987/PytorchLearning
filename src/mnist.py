"""
案例:Pytorch搭建全连接网络，实现MNIST手写识别
1使用DataLoader加载数据集
2定义网络模型，选择损失函数和优化器
3在训练集上训练模型
4使用测试集评估模型
5识别指定图片
"""

import torch
import matplotlib

matplotlib.use('TkAgg')  # 设置后端为TkAgg
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
# 创建DataLoader对象
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # shuffle=True表示随机打乱顺序
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # shuffle=False表示不打乱顺序

# 使用Matplotlib可视化手写数字图片
# 查看训练集中前10张图片
fig = plt.figure(figsize=(9, 4))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.matshow(train_dataset[i][0].reshape(28, 28))
plt.tight_layout()  # 自动调整子图参数
plt.show()


# 建立三层全连接网络，并在线性层和非线性层之间增加批标准化以加快收敛速度
class My_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(My_Net, self).__init__()

        self.fc1 = nn.Linear(in_dim, n_hidden_1)  # 第1个线性层
        self.bn1 = nn.BatchNorm1d(n_hidden_1)  # 第1个批标准化层 加快收敛速度

        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)  # 第2个线性层
        self.bn2 = nn.BatchNorm1d(n_hidden_2)  # 第2个批标准化层
        self.out = nn.Linear(n_hidden_2, out_dim)  # 输出层

        self.layer1 = nn.Sequential(self.fc1, self.bn1, nn.ReLU(True))  # inplace=True表示操作会直接在原始数据上进行修改，而不是创建新的输出
        self.layer2 = nn.Sequential(self.fc2, self.bn2, nn.ReLU(True))
        self.layer3 = nn.Sequential(self.out)

    # 定义前向传播过程
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 创建模型对象，选择损失函数和优化器
model = My_Net(in_dim, n_hidden_1, n_hidden_2, out_dim)  # 创建模型对象
if torch.cuda.is_available():  # 判断是否有GPU
    print("使用GPU加速")
    model = model.cuda()  # 将模型加载到GPU上

myloss = nn.CrossEntropyLoss()  # 选择交叉熵损失函数

# 创建优化器对象
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam作为优化器

# 迭代训练模型
for epoch in range(epoches):
    for (i, data) in enumerate(train_loader):
        img, label = data
        img = img.view(img.size(0), -1)  # 将图片展平为一维向量
        if torch.cuda.is_available():  # 判断是否有GPU
            # print("训练——使用GPU加速")
            img = img.cuda()  # 将图片加载到GPU上
            label = label.cuda()  # 将标签加载到GPU上
        else:
            img = Variable(img)  # 将图片转换为Variable对象
            label = Variable(label)  # 将标签转换为Variable对象
        # 将数据交给模型对象，前向传播，计算损失（误差）
        out = model(img)
        loss = myloss(out, label)
        # 反向传播，计算梯度，更新参数
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        # 输出当前训练损失
        print(f"Epoch{epoch}/{epoches},loss:{loss.data.item():.8f}")

# 使用测试集评估模型
model.eval()  # 将模型设置为评估模式
eval_loss = 0  # 训练集损失
eval_acc = 0  # 训练集准确率
for (i, data) in enumerate(test_loader):
    img, label = data
    img = img.view(img.size(0), -1)  # 将图片展平为一维向量
    if torch.cuda.is_available():  # 判断是否有GPU
        # print("评估——使用GPU加速")
        img = img.cuda()  # 将图片加载到GPU上
        label = label.cuda()  # 将标签加载到GPU上
    else:
        img = Variable(img)  # 将图片转换为Variable对象
        label = Variable(label)  # 将标签转换为Variable对象

    # 将数据交给模型对象，前向传播，计算损失（误差）
    out = model(img)
    loss = myloss(out, label)

    # 计算损失：将当前批次损失值乘以批次中样本数量，累加到总损失中
    eval_loss += loss.data.item() * label.size(0)  # 计算损失
    _, pred = torch.max(out, 1)  # 获取预测结果中每行最大值的索引，即预测类别
    # 计算预测正确的样本数量
    # 使用pred和label进行逐元素比较，得到一个布尔张量
    # True表示预测正确，False表示预测错误
    # 使用sum()函数统计True的总数，即预测正确的样本数量
    num_correct = (pred == label).sum()  # 计算预测正确的样本数量
    eval_acc += num_correct.item()  # 计算预测正确的样本数量

# 输出评估结果
print(f"Test Loss: {eval_loss / len(test_dataset):.4f}, Test Accuracy: {eval_acc / len(test_dataset):.4f}")

# 识别测试集中的指定图片
idx = 0  # 指定要识别的图片索引
img, label = test_dataset[idx]  # 获取指定索引的图片和标签
img = img.view(img.size(0), -1)  # 将图片展平为一维向量
if torch.cuda.is_available():  # 判断是否有GPU
    img = img.cuda()  # 将图片加载到GPU上
else:
    img = Variable(img)  # 将图片转换为Variable对象

# 向前传播
out = model(img)
_, pred = torch.max(out, 1)  # 获取预测结果中每行最大值的索引，即预测类别
# 输出预测结果
print(f"Predicted class: {pred.item()}, True class: {label}")
plt.matshow(img.cpu().reshape(28, 28))  # 显示图片
plt.show()  # 显示图片
