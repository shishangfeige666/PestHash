import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
import torchvision.datasets as dsets
from tqdm import tqdm


def config_dataset(config):
    if 'CUB' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 200
    elif 'nabirds' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 555
    elif 'car' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 196
    elif 'APTV99' in config['dataset']:
        config['topk'] = -1
        config['n_class'] = 99
    # elif 'imagenet' in config['dataset']:
    #     config['topk'] = 1000
    #     config['n_class'] = 100
    elif 'coco' in config['dataset']:
        config['topk'] = 5000
        config['n_class'] = 80
    elif 'imagenet1k' in config['dataset']:
        config['topk'] = 1000
        config['n_class'] = 1000
    elif 'cifar' in config["dataset"]:
        config["topk"] = -1
        config["n_class"] = 10
    elif 'ip102' in config["dataset"]:
        config["topk"] = -1
        config["n_class"] = 102



    if config['dataset'] == 'CUB_200_2011':
        config['data_path'] = '../../data/' + config['dataset'] + '/images/'
    if config['dataset'] == 'nabirds':
        config['data_path'] = '../../data/' + config['dataset'] + '/images/'

    if config['dataset'] == 'car_ims':
        config['data_path'] = '../../data/stanford_cars/'
    if config['dataset'] == 'APTV99':
        config['data_path'] = '../../data/APTV99-publish/'
    if config['dataset'] == 'ip102':
        config['data_path'] = '../../data/ip102_v1.1/images/'

    if config['dataset'] == 'imagenet':
        config['data_path'] = '../../data/imagenet/'
    if config['dataset'] == 'mscoco':
        config['data_path'] = '../../data/coco/'
    if config['dataset'] == 'imagenet1k':
        config['data_path'] = '../../data/'
    
    if config['dataset'] == 'imagenet' or config['dataset'] == 'mscoco' or config['dataset'] == 'imagenet1k':
        config['data'] = {
            'train_set': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
            'test_set': {'list_path': './data/' + config['dataset'] + '/test.txt', 'batch_size': config['batch_size']},
            'database': {'list_path': './data/' + config['dataset'] + '/database.txt', 'batch_size': config['batch_size']},
        }
    else:
        config['data'] = {
            'train_set': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
            'test_set': {'list_path': './data/' + config['dataset'] + '/test.txt', 'batch_size': config['batch_size']},
            'database': {'list_path': './data/' + config['dataset'] + '/train.txt', 'batch_size': config['batch_size']},
        }

    return config


def encode_onehot(labels, num_classes=10):
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


class ImageList(object):
    def __init__(self, data_path, image_list, transform, n_class):
        self.imgs = [data_path + val.strip().split('\t')[0] for val in image_list] #ip102为[int(val.strip().split(' ')[1]) for val in image_list]，其他数据集为[int(val.strip().split('\t')[1]) for val in image_list]
        self.transform = transform
        self.labels = [int(val.strip().split('\t')[1]) for val in image_list] #ip102为[int(val.strip().split(' ')[1]) for val in image_list]，其他数据集为[int(val.strip().split('\t')[1]) for val in image_list]
        self.n_class = n_class

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        target = np.eye(self.n_class, dtype=np.float32)[np.array(target)]
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def get_one_hot_label(self):
        return torch.from_numpy(encode_onehot(self.labels, self.n_class)).float()

class ImageNet(object):
    def __init__(self, data_path, image_list, transform, n_class):
        self.imgs = [(data_path + val.strip().split()[0], np.array([float(la) for la in val.strip().split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        target = target.astype(np.float32)
        return img, target, index

    def __len__(self):
        return len(self.imgs)




def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)


    dsets = {}
    dset_loaders = {}
    data_config = config['data']

    for data_set in ['train_set', 'test_set', 'database']:
        dsets[data_set] = ImageList(config['data_path'],
                                    open(data_config[data_set]['list_path']).readlines(),
                                    transform=image_transform(config['resize_size'], config['crop_size'], data_set),
                                    n_class=config['n_class'])
        print(data_set, len(dsets[data_set]))
        if data_set == 'train_set':
            isShuffle = True
        else:
            isShuffle = False
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]['batch_size'],
                                                      shuffle=isShuffle, num_workers=4)
    return dset_loaders['train_set'], dset_loaders['test_set'], dset_loaders['database'], \
           len(dsets['train_set']), len(dsets['test_set']), len(dsets['database'])

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]  #np.eye(10, dtype=np.int8)对角线全为1，其余为0的10*10的矩阵
        return img, target, index

def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(root='D:/AppData/python/PyCharmProjects/data/cifar',
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root='D:/AppData/python/PyCharmProjects/data/cifar',
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root='D:/AppData/python/PyCharmProjects/data/cifar',
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))  # 拼接训练集和测试集
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))  # 拼接标签和测试标签成一个numpy数组

    first = True
    for label in range(10):
        index = np.where(L == label)[0]  # label==targets
        N = index.shape[0]  # shape[0]是index的长度                     shape[0]图像的高度，shape[1]图像的宽度，shape[2]图像的通道数
        perm = np.random.permutation(N)  # 随机排序N
        index = index[perm]  # 对index随机排序

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,  # true
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,  # False
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,  # False
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
        train_index.shape[0], test_index.shape[0], database_index.shape[0]

def get_imagenet_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test_set", "database"]:
        dsets[data_set] = ImageNet(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set),
                                    n_class=config['n_class'])
        print(data_set, len(dsets[data_set]))
        if data_set == 'train_set':
            isShuffle = True
        else:
            isShuffle = False
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=isShuffle, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test_set"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test_set"]), len(dsets["database"])


if __name__ == '__main__':
    config = get_config()
    train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    for img, label, ind in train_loader:
        print(img.shape)
        print(label.shape)
        break
