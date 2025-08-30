import pandas as pd
from torch.utils.data import Dataset


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __getitem__(self, idx):  # 根据索引获取数据
        data = (self.csv_data[idx], self.txt_data[idx])
        return data

    def __len__(self):  # 返回数据集长度
        # 返回数据集的大小
        return len(self.csv_data)


def dataloader_demo():
    from torch.utils.data import DataLoader
    # 定义数据加载器
    # 常用参数：dataset(加载数据的数据集)、batch_size(每个batch加载多少个样本)shuffle(是否随机打乱顺序)、num_workers(使用多少个子线程加载数据)
    dataiter = DataLoader(dataset, batch_size=32, shuffle=True)


def image_folder_demo():
    from torchvision.datasets import ImageFolder
    # 定义加载器
    imgset = ImageFolder(root='../static/cats_dogs', transform=None)  # transform=None表示不进行任何转换
    # 查看数据信息
    print(imgset.classes)  # 查看类别,目录名称
    print(imgset.class_to_idx)  # 查看类别对应的索引
    print(imgset.imgs)  # 查看所有图片的路径和标签


if __name__ == '__main__':
    image_folder_demo()
