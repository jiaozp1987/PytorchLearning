import numpy


def main():
    # Tensor 的定义和操作
    import torch
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(a)
    print(a.shape)
    print(a.size())  # 输出形状
    print(len(a))
    b = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
    print(b)
    print(b.shape)
    # 定义全0或全1的tensor
    c = torch.zeros(3, 4)  # 3行4列
    print(c)
    d = torch.ones(3, 4)
    print(d)
    # 指定的数值的tensor
    e = torch.Tensor(3, 4).fill_(10)
    print(e)
    e = torch.full((3, 4), 10)
    print(e)
    # 定义随机tensor
    f = torch.rand(3, 4)  # 默认0-1之间
    print(f)
    f = torch.randn(3, 4)  # 符合正态分布的随机数
    print(f)
    # 取值、赋值、切片
    print(a[0, 1])  # 取值
    a[0, 1] = 100
    print(a[0, 1])
    a_slice_columns = a[:, -1]
    print(a_slice_columns)
    a_slice_rows = a[1, :]
    print(a_slice_rows)

    # tensor 与numpy转换
    import numpy as np
    # torch.Tensor -> numpy.ndarray
    numpy_a = a.numpy()
    print(numpy_a)
    print(numpy_a.shape)
    print(numpy_a.size)
    # numpy.ndarray -> torch.Tensor
    g = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
    print(g)
    torch_g = torch.from_numpy(g)
    print(torch_g)


if __name__ == '__main__':
    main()
