import os, random, ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.68660736, 0.56024116, 0.43283942], std=[0.18383975, 0.16505268, 0.15739757])
])

# 按数量划分
def shufff(path):
    """垃圾 os"""
    path1 = os.path.basename(path)
    dataset = []
    name = []
    for person in os.listdir(path1):  # 遍历文件夹
        name.append(person)
        for imgname in os.listdir(os.path.abspath(os.path.join(path1, person))):
            dataset.append([os.path.join(path, person, imgname), person])
            # self.dataset.append([os.path.abspath(os.path.join(path, person, imgname)), person])
    # ----------划分训练验证----------#
    file_train = open('train.txt', 'w')
    file_val = open('val.txt', 'w')
    train_percent = 0.9
    val_percent = 0.1

    num = len(dataset)  # 910
    print('总的数据量：', num)
    list_index = range(num)  # (0,910)
    train_num = int(num * train_percent)  # 819
    print('划分训练集的数量：', train_num)
    val_num = int(num * val_percent)  # 91
    print('划分验证集的数量：', val_num)

    train = random.sample(list_index, train_num)  # 从910个随机选择819个
    val = random.sample(list_index, val_num)
    for i in range(num):
        if i in train:
            file_train.write(str(dataset[i]) + '\n')
        else:
            file_val.write(str(dataset[i]) + '\n')
    file_train.close()
    file_val.close()
    # 上面写入的太整齐，将其打乱
    with open('train.txt', 'r') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open('train.txt', 'w') as file:
        file.writelines(lines)
    #
    with open('val.txt', 'r') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open('val.txt', 'w') as file:
        file.writelines(lines)

# 按类别划分
def shuff(path):
    """垃圾 os"""
    path1 = os.path.basename(path)
    dataset = []
    name = []
    for person in os.listdir(path1):  # 遍历文件夹
        name.append(person)
        for imgname in os.listdir(os.path.abspath(os.path.join(path1, person))):
            dataset.append([os.path.join(path, person, imgname), person])
            # self.dataset.append([os.path.abspath(os.path.join(path, person, imgname)), person])
    # ----------划分训练验证----------#
    file_train = open('train.txt', 'w')
    file_val = open('val.txt', 'w')
    train_percent = 0.9
    val_percent = 0.1

    num = len(dataset)  # 910
    print('总的数据量：', num)
    list_index = range(num)  # (0,910)
    train_num = int(num * train_percent)  # 819
    print('划分训练集的数量：', train_num)
    val_num = int(num * val_percent)  # 91
    print('划分验证集的数量：', val_num)

    train = random.sample(list_index, train_num)  # 从910个随机选择819个
    val = random.sample(list_index, val_num)
    for i in range(num):
        if i in train:
            file_train.write(str(dataset[i]) + '\n')
        else:
            file_val.write(str(dataset[i]) + '\n')
    file_train.close()
    file_val.close()
    # 上面写入的太整齐，将其打乱
    with open('train.txt', 'r') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open('train.txt', 'w') as file:
        file.writelines(lines)
    #
    with open('val.txt', 'r') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open('val.txt', 'w') as file:
        file.writelines(lines)
class My_Data(Dataset):
    def __init__(self, path, get_name_list=r'datas/train_val'):
        # 传入train.txt 和数据集路径，也可以从上面shufff中取出name来
        self.name = []
        for person in os.listdir(get_name_list):  # 遍历文件夹
            self.name.append(person)
        with open(path) as f:
            self.dataset = f.readlines()
        # self.__getitem__(5)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        data = ast.literal_eval(data.strip())
        img_data = Image.open(data[0])
        img_data = img_data.resize((224, 224))
        label = self.name.index(data[1])
        return transform(img_data), label  # 处理后的图像，和名字


if __name__ == '__main__':
    # shufff('datas/train_val')
    # x=My_Data(r'train.txt')
    shuff()
