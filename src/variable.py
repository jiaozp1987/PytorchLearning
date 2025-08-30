import torch
from torch.autograd import Variable


def automatic_guidance():
    # 创建三个变量
    x = Variable(torch.Tensor([1]), requires_grad=True)  # requires_grad=True表示需要计算梯度(求导)
    w = Variable(torch.Tensor([2]), requires_grad=True)
    b = Variable(torch.Tensor([3]), requires_grad=True)
    # 创建一个计算图
    y = w * x + b
    print(y)
    # 自动求导
    y.backward()  # 反向求导
    print(x.grad, w.grad, b.grad)


def matrix_derivative():
    x = torch.randn(3)
    print(x)
    x = Variable(x, requires_grad=True)
    print(x)
    # 定义计算题
    y = 6 * x
    print(y)
    # 自动求导
    y.backward(torch.FloatTensor([1, 0.1, 0.01]))  # 反向求导
    print(x.grad)


# 叶子节点的梯度在反向传播后会被保留
# 叶节点：由用户直接创建的计算图Variable对象
def leaf():
    a = Variable(torch.ones(3, 4), requires_grad=True)
    b = Variable(torch.zeros(3, 4), requires_grad=False)
    c = a.add(b)  # c依赖a和b
    d = c.sum() # d依赖c
    print(d)
    d.backward()
    print(a.requires_grad, b.requires_grad, c.requires_grad, d.requires_grad)


if __name__ == '__main__':
    # automatic_guidance()
    # matrix_derivative()
    leaf()
