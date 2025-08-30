import torch
from torch import nn

"""
nn.Module 简单的代码示意，代码不一定能跑通
"""

class MyNet(nn.Module):
    """
    定义神经网络实例（2个卷积层）
    """
    def __init__(self):
        """
        初始化函数，用于创建神经网络层
        """

        super(MyNet, self).__init__()  # 调用父类的初始化方法，确保继承自nn.Module的类正确初始化
        # 创建一个二维卷积层
        # 输入参数说明：
        # 1 - 输入通道数（维度）（因为是灰度图像，所以为1）
        # 20 - 输出通道数（维度）（卷积产生的特征图数量）
        # 5 - 卷积核的大小（5x5的卷积窗口）
        self.conv1 = nn.Conv2d(1, 20, 5)
        # 创建一个二维卷积层
        # 输入通道数为20
        # 输出通道数为20
        # 卷积核大小为5x5
        self.conv2 = nn.Conv2d(20, 20, 5)

        # self.fc = nn.Linear(10, 1)  # 定义一个全连接层，输入维度为10，输出维度为1

    def forward(self, x): # x为输入数据，通常是特征张量 Variable
        x = self.relu(self.conv1(x))  # 将输入数据传入卷积层1，并使用ReLU激活函数
        x = self.relu(self.conv2(x))  # 将卷积层1的输出传入卷积层2，并使用ReLU激活函数
        return x



    def loss_func(self):
        """
        定义和使用损失函数
        :return:
        """
        # 导入并定义均方误差损失函数，设置reduction为"sum"表示将所有误差相加
        loss_fun = nn.MSELoss(reduction="sum")
        # 生成一个包含5个元素的随机张量作为模型的预测输出
        y_pred = torch.rand(5)
        # 创建一个全1的张量作为真实值，维度与预测输出相同
        y = torch.ones(5)
        # 计算预测值与真实值之间的均方误差损失
        loss = loss_fun(y_pred, y)
        print(loss)

    def optim(self):
        """
        定义和使用优化器
        :return:
        """
        import torch.optim as optim
        # 定义优化器(参数：网络模型的参数，学习率)
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降（SGD）作为优化器 model.parameters()表示网络模型的参数
        #使用过程（示意）
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

    def model_save_load(self):
        # 保存全模型
        torch.save(model, 'model.pth') # model 是模型对象

        # 只保存参数状态
        torch.save(model.state_dict(), 'model.pth') # model 是模型对象

        # 加载全模型
        model = torch.load('model.pth')

        # 加载参数状态
        model.load_state_dict(torch.load('model.pth')) # model 是模型对象

