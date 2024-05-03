from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.util import download_url, check_integrity, noisify, noisify_instance

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, Ba_num, sample_ratio, noise_mode, noise_rate,  noise_type, eps, root_dir,result_dir, transform,
                 mode, noise_file='', probability=[], log='', random_state=0):

        self.noise_rate = noise_rate
        self.transform = transform
        self.mode = mode
        self.result_dir_save = result_dir
        self.sample_ratio = sample_ratio
        self.class_ind = {}
        self.num_sample = 50000
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
                self.nb_classes = 10
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
                self.nb_classes = 100
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                self.train_label1 = train_label
                train_data = np.concatenate(train_data)
                self.nb_classes = 10
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                self.nb_classes = 100
            self.train_label1 = train_label
            idx_each_class_noisy = [[] for i in range(self.nb_classes)]
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                print(noise_file)
                noise_label = json.load(open(noise_file, "r"))
                for kk in range(self.nb_classes):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]
            else: 
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.noise_rate * 50000)
                noise_idx = idx[:num_noise]

                if noise_mode != 'instance':
                    train_label = np.asarray([[train_label[i]] for i in range(len(train_label))])
                    noise_label1, self.actual_noise_rate = noisify(dataset=dataset, train_labels=train_label,
                                                                   noise_type=noise_mode, noise_rate=self.noise_rate,
                                                                   random_state=random_state,
                                                                   nb_classes=self.nb_classes)

                    for i in range(50000):
                        noise_label.append(int(noise_label1[i]))

                    noise_label1 = [i[0] for i in noise_label1]
                    _train_labels = [i[0] for i in train_label]
                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[noise_label1[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(noise_label1) != np.transpose(_train_labels)

                else:
                    noise_label1, self.actual_noise_rate = noisify_instance(train_data, train_label,
                                                                            noise_rate=self.noise_rate)

                    print('over all noise rate is ', self.actual_noise_rate)
                    for i in range(50000):
                        noise_label.append(int(noise_label1[i]))

                    noise_label1 = [i for i in noise_label1]
                    _train_labels = [i for i in train_label]

                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[noise_label1[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(noise_label1) != np.transpose(_train_labels)
                print("save noisy labels to %s ..." % noise_file)
                json.dump((noise_label), open(noise_file, "w"))
                for kk in range(self.nb_classes):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            if self.mode == 'all' or self.mode == 'train_all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                save_file = 'Clean_index_'+ str(dataset) + '_' + noise_type + '_' + str(eps) + '_' + noise_mode + '_' + str(self.noise_rate) + '.json'
                save_file = os.path.join(self.result_dir_save, save_file)
                if self.mode == "labeled":
                    pred_idx1 = list([])
                    class_len = int(self.sample_ratio*self.num_sample/self.nb_classes)

                    clean = (np.array(noise_label) == np.array(train_label))
                    size_pred = 0
                    for i in range(self.nb_classes):
                        class_indices = self.class_ind[i]
                        prob_  = np.argsort(probability[class_indices].cpu().numpy())
                        size1 = len(class_indices)

                        try:
                            pidx = np.array(class_indices)[prob_[0:int(Ba_num[i])].astype(int)].squeeze()
                            pred_idx1.extend(pidx)
                        except:
                            pidx = np.array(class_indices).squeeze()
                            pred_idx1.extend(pidx)
                    pred_idx = [int(x) for x in list(pred_idx1)]
                    json.dump((pred_idx), open(save_file, "w"))

                elif self.mode == "unlabeled":
                    pred_idx = json.load(open(save_file, "r"))
                    idx = list(range(self.num_sample))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]        
                    pred_idx = pred_idx_noisy   

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            img4 = self.transform[3](img)
            return img1, img2, img3, img4, target, prob
        elif self.mode == 'unlabeled':
            img, target  = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            img4 = self.transform[3](img)
            return img1, img2, img3, img4, target 
        elif self.mode == 'all':
            img, target, target1 = self.train_data[index], self.noise_label[index], self.train_label1[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, target1, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_poisoneddataset(Dataset):
    def __init__(self, dataset, Ba_num, sample_ratio, noise_mode, noise_rate, eps, root_dir, test_dir, result_dir, constraint, poison_type, transform,
                 mode, noise_file='', probability=[], log='', random_state=0):

        self.noise_rate = noise_rate 
        self.transform = transform
        self.constraint = constraint
        self.poison_type = poison_type
        self.mode = mode
        self.result_dir_save = result_dir
        self.root = os.path.expanduser(root_dir)
        self.test_dir = os.path.expanduser(test_dir)
        #path for poisoned training data
        self.file_path = os.path.join(self.root, '{}.{}'.format(self.constraint, self.poison_type.lower()))
        #path for poisoned test data
        self.testfile_path = os.path.join(self.test_dir, '{}.{}'.format(self.constraint, self.poison_type.lower()))
        self.sample_ratio = sample_ratio
        self.class_ind = {}
        self.num_sample = 50000
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_data, self.test_label = torch.load(self.testfile_path)
                self.nb_classes = 10
            elif dataset == 'cifar100':
                test_data, self.test_label = torch.load(self.testfile_path)
                self.nb_classes = 100
            self.test_data = test_data.permute(0, 2, 3, 1)   # convert to HWC
            self.test_data = (self.test_data * 255).type(torch.uint8)
        elif self.mode == 'train_tsne':
            self.train_data = []
            self.train_label_tsne = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    self.train_data.append(data_dic['data'])
                    self.train_label_tsne = self.train_label_tsne + data_dic['labels']
                self.train_data = np.concatenate(self.train_data)
                self.nb_classes = 10
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                self.train_data = train_dic['data']
                self.train_label_tsne = train_dic['fine_labels']
                self.nb_classes = 100
            idx_each_class_noisy = [[] for i in range(self.nb_classes)]
            train_data = train_data.permute(0, 2, 3, 1)   # convert to HWC
        else:
            train_data = []
            train_label = []
            train_data, train_label = torch.load(self.file_path)
            if dataset == 'cifar10':
                self.nb_classes = 10
            elif dataset == 'cifar100':
                self.nb_classes = 100
            self.train_label1 = train_label
            idx_each_class_noisy = [[] for i in range(self.nb_classes)]
            train_data = train_data.permute(0, 2, 3, 1)   # convert to HWC
            train_data = (train_data * 255).type(torch.uint8)
            if os.path.exists(noise_file):
                print(noise_file)
                noise_label = json.load(open(noise_file, "r"))
                for kk in range(self.nb_classes):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]
            else: 
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.noise_rate * 50000)
                noise_idx = idx[:num_noise]

                if noise_mode != 'instance':
                    train_label = np.asarray([[train_label[i]] for i in range(len(train_label))])
                    noise_label1, self.actual_noise_rate = noisify(dataset=dataset, train_labels=train_label,
                                                                   noise_type=noise_mode, noise_rate=self.noise_rate,
                                                                   random_state=random_state,
                                                                   nb_classes=self.nb_classes)

                    for i in range(50000):
                        noise_label.append(int(noise_label1[i]))

                    noise_label1 = [i[0] for i in noise_label1]
                    _train_labels = [i[0] for i in train_label]
                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[noise_label1[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(noise_label1) != np.transpose(_train_labels)

                else:
                    noise_label1, self.actual_noise_rate = noisify_instance(train_data, train_label,
                                                                            noise_rate=self.noise_rate)

                    print('over all noise rate is ', self.actual_noise_rate)
                    for i in range(50000):
                        noise_label.append(int(noise_label1[i]))

                    noise_label1 = [i for i in noise_label1]
                    _train_labels = [i for i in train_label]

                    for i in range(len(_train_labels)):
                        idx_each_class_noisy[noise_label1[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(noise_label1) != np.transpose(_train_labels)
                print("save noisy labels to %s ..." % noise_file)
                json.dump((noise_label), open(noise_file, "w"))
                for kk in range(self.nb_classes):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            if self.mode == 'all' or self.mode == 'train_all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                save_file = 'Clean_index_'+ str(dataset) + '_' + poison_type + '_' + str(eps) + '_' + noise_mode + '_' + str(self.noise_rate) + '.json'
                save_file = os.path.join(self.result_dir_save, save_file)
                if self.mode == "labeled":
                    pred_idx1 = list([])
                    class_len = int(self.sample_ratio*self.num_sample/self.nb_classes)

                    clean = (np.array(noise_label) == np.array(train_label))
                    size_pred = 0
                    for i in range(self.nb_classes):
                        class_indices = self.class_ind[i]
                        prob_  = np.argsort(probability[class_indices].cpu().numpy())
                        size1 = len(class_indices)

                        try:
                            pidx = np.array(class_indices)[prob_[0:int(Ba_num[i])].astype(int)].squeeze()
                            pred_idx1.extend(pidx)
                        except:                            
                            pidx = np.array(class_indices).squeeze()
                            pred_idx1.extend(pidx)

                    pred_idx = [int(x) for x in list(pred_idx1)]
                    json.dump((pred_idx), open(save_file, "w"))

                elif self.mode == "unlabeled":
                    pred_idx = json.load(open(save_file, "r"))
                    idx = list(range(self.num_sample))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]        
                    pred_idx = pred_idx_noisy   

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img.numpy())
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            img4 = self.transform[3](img)
            return img1, img2, img3, img4, target, prob
        elif self.mode == 'unlabeled':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img.numpy())
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            img4 = self.transform[3](img)
            return img1, img2, img3, img4, target
        elif self.mode == 'all':
            img, target, target1 = self.train_data[index], self.noise_label[index], self.train_label1[index]
            img = Image.fromarray(img.numpy())
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, target1, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img.numpy())
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, noise_mode, noise_rate, noise_type, eps, batch_size, num_workers, root_dir, result_dir, log, noise_file=''):
        self.dataset = dataset
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.eps = eps
        self.result_dir = result_dir
        if self.dataset == 'cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                ]
            )
            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])

        elif self.dataset == 'cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
                "labeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
            }
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

    def run(self, sample_ratio, mode, Ba_num=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                        noise_rate=self.noise_rate, noise_type = self.noise_type, eps = self.eps, root_dir=self.root_dir,result_dir=self.result_dir,
                                        transform=self.transforms["warmup"], mode="all", noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                            noise_rate=self.noise_rate, noise_type = self.noise_type, eps = self.eps, root_dir=self.root_dir,result_dir=self.result_dir,
                                            transform=self.transforms["labeled"], mode="labeled", noise_file=self.noise_file, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                              noise_rate=self.noise_rate, noise_type = self.noise_type, eps = self.eps, root_dir=self.root_dir,result_dir=self.result_dir,
                                              transform=self.transforms["unlabeled"], mode="unlabeled", noise_file=self.noise_file)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                         noise_rate=self.noise_rate, noise_type = self.noise_type, eps = self.eps, root_dir=self.root_dir,result_dir=self.result_dir,
                                         transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                         noise_rate=self.noise_rate, noise_type = self.noise_type, eps = self.eps, root_dir=self.root_dir,result_dir=self.result_dir,
                                         transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

class cifar_poisoneddataloader():
    def __init__(self, dataset, noise_mode, noise_rate, eps, batch_size, num_workers, root_dir, test_dir, result_dir, constraint, poison_type, log, noise_file=''):               
        self.dataset = dataset
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.poison_type = poison_type
        self.constraint = constraint
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.log = log
        self.noise_file = noise_file
        self.noise_rate = noise_rate
        self.eps = eps
        self.result_dir = result_dir
        if self.dataset == 'cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])

        elif self.dataset == 'cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
                "labeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
            }
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

    def run(self, sample_ratio, mode, Ba_num=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_poisoneddataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                        noise_rate=self.noise_rate, eps = self.eps, root_dir=self.root_dir, test_dir=self.test_dir, result_dir=self.result_dir, constraint= self.constraint, poison_type= self.poison_type,
                                        transform=self.transforms["warmup"], mode="all", noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_poisoneddataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                            noise_rate=self.noise_rate, eps = self.eps, root_dir=self.root_dir, test_dir=self.test_dir,result_dir=self.result_dir, constraint= self.constraint, poison_type= self.poison_type,
                                            transform=self.transforms["labeled"], mode="labeled", noise_file=self.noise_file,
                                            probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)

            unlabeled_dataset = cifar_poisoneddataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                              noise_rate=self.noise_rate, eps = self.eps, root_dir=self.root_dir, test_dir=self.test_dir,result_dir=self.result_dir, constraint= self.constraint, poison_type= self.poison_type,
                                              transform=self.transforms["unlabeled"],
                                              mode="unlabeled", noise_file=self.noise_file)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_poisoneddataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                         noise_rate=self.noise_rate, eps = self.eps, root_dir=self.root_dir, test_dir = self.test_dir, result_dir=self.result_dir, constraint= self.constraint, poison_type= self.poison_type,
                                         transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'eval_train':
            eval_dataset = cifar_poisoneddataset(dataset=self.dataset, Ba_num=Ba_num, sample_ratio= sample_ratio, noise_mode=self.noise_mode,
                                         noise_rate=self.noise_rate, eps = self.eps, root_dir=self.root_dir, test_dir=self.test_dir ,result_dir=self.result_dir, constraint= self.constraint, poison_type= self.poison_type,
                                         transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
            