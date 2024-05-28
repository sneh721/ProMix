import os
import torch
import copy
import random
import json
from utils.randaug import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision import datasets
import matplotlib.pyplot as plt


class cifarn_dataset(Dataset):
    def __init__(self,  dataset,  noise_type, noise_path, root_dir, transform, mode, transform_s=None, is_human=False, noise_file='',
                 pred=[], probability=[],probability2=[] ,log='', print_show=False, r =0.2 , noise_mode = 'sym'):
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.print_show = print_show
        self.noise_mode = noise_mode
        self.r = r
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        # self.train_noisy_labels -- training set of noise labels

        # Our dataset has two classes == horse, cow
        self.nb_classes = 2
        idx_each_class_noisy = [[] for i in range(self.nb_classes)]

        if self.mode == 'test':
            test_dataset = datasets.ImageFolder(root='Images/valid', transform=self.transform)
            self.test_data = []
            self.test_label = []
            for img, label in test_dataset:
                img = (img.numpy() * 255).astype(np.uint8)   # Convert 0 to 1 pixels to 0 to 255
                img = np.transpose(img, (1, 2, 0))          # (H, W, C)
                self.test_data.append(img)  # Convert the image tensor to a numpy array
                self.test_label.append(label)
        else:   # Train or warmup mode
            # Without any transformation, ImageFolder returns a Image type (RGB)
            # With transform -- since we do toTensor(), returns images with each pixel 0 to 1
            # It is easier to do this way to easily get labels and also do the same things in test
            train_dataset = datasets.ImageFolder(root='Images/train', transform=self.transform)
            train_data = []
            train_label = []
            for img, label in train_dataset:
                img = (img.numpy() * 255).astype(np.uint8)    # Convert 0 to 1 pixels to 0 to 255
                img = np.transpose(img, (1, 2, 0))
                train_data.append(img)  # Convert the image tensor to a numpy array
                train_label.append(label)
            self.train_label = train_label

            # NOTE: the code below seems to be for testing: basically either you already know the noise labels
            # OR the data is clean and we inject noise
            # In real life case: the train labels are noisy already and we don't know the noise pattern
            # if noise_type is not None:
            if os.path.exists(noise_file):           # Seems like the noise file has the noisy label: 
                noise_label = json.load(open(noise_file,"r"))
                self.train_noisy_labels = noise_label
                # creates a boolean array that identifies which labels are noisy by comparing them with the true labels
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_label)
            else:    #inject noise   
                # sym = symmetric noise: The noise is distributed uniformly across all classes
                # asym = asymmetric noise: more realistic scenarios where certain classes are more 
                # likely to be confused with particular other classes
                if self.noise_mode=='sym' or self.noise_mode =='asym':
                    noise_label = []
                    n = len(train_dataset)
                    idx = list(range(n))
                    random.shuffle(idx)
                    num_noise = int(self.r * n)            
                    noise_idx = idx[:num_noise]
                    for i in range(n):
                        if i in noise_idx:
                            if self.noise_mode=='sym':
                                noiselabel = random.randint(self.nb_classes)
                                noise_label.append(noiselabel)
                            elif self.noise_mode=='asym':
                                noiselabel = self.transition[self.train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(self.train_label[i])
                    self.train_noisy_labels = noise_label
                    self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_label)
                    print("save noisy labels to %s ..."%noise_file)        
                    json.dump(noise_label,open(noise_file,"w"))
                elif self.noise_mode == 'custom':                     # Branch for data already is noisy
                    if noise_type != 'clean':
                        self.train_noisy_labels = self.train_label
                    
                        for i in range(len(self.train_noisy_labels)):
                            idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                        class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
                        self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                        self.print_wrapper(f'The noisy data ratio in each class is {self.noise_prior}')
                        self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_label)
                        self.actual_noise_rate = np.sum(self.noise_or_not) / len(self.noise_or_not)
                        self.print_wrapper('over all noise rate is ', self.actual_noise_rate)
                    noise_label = self.train_noisy_labels
                
            
            if self.mode == 'all_lab':   # fully labeled data with probability information
                self.probability = probability
                self.probability2 = probability2
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':     # fully labeled data without probability information
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":     
                    # Data labeled by predictions (pseudo labeled): using the probability distribution of the predictions of data
                    # Then checks the cleanness of the pseudo label
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))    # Not sure why this is here
                    log.write('Numer of labeled samples:%d   AUC (not computed):%.3f\n' % (pred.sum(), 0))
                    log.flush()

                elif self.mode == "unlabeled":        # Unlabeled data -- just model predictions
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.print_wrapper("%s data has a size of %d" % (self.mode, len(self.noise_label)))
        self.print_show = False
    
    # Just for printing stuff
    def print_wrapper(self, *args, **kwargs):
        if self.print_show:
            print(*args, **kwargs)

    def load_label(self):
        # NOTE only load manual training label -- not sure what they mean by "manual"
        # handling noisy and clean labels for a training dataset 
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0
                self.print_wrapper(f'Loaded {self.noise_type} from {self.noise_path}.')
                self.print_wrapper(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img, target, prob, prob2 = self.train_data[index], self.noise_label[index], self.probability[index],self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob,prob2,true_labels, index
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]   # img => numpy array, target => int
            img = Image.fromarray(img)
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':     # What is all vs all2????
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
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

class RandomAugmentToTensor(transforms.Compose):
    def __call__(self, img):
        img = super().__call__(img)  # Apply all transformations except ToTensor
        # Convert PIL image to tensor
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_tensor

class cifarn_dataloader():
    def __init__(self, dataset, noise_type, noise_path, is_human, batch_size, num_workers, root_dir, log,
                 noise_file='',noise_mode='custom', r=0.2):
        self.r = r
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.is_human = is_human
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'custom':
            self.transform_train = transforms.Compose([
                                        transforms.CenterCrop((240, 300)),           # Hardcoded because I cropped the images
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(1, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.CenterCrop((240, 300)),
                transforms.ToTensor(),
            ])
        self.print_show = True

    def run(self, mode, pred=[], prob=[],prob2=[]):
        if mode == 'warmup':
            all_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                         is_human=self.is_human, root_dir=self.root_dir, transform=self.transform_train,
                                         transform_s=self.transform_train_s, mode="all",
                                         noise_file=self.noise_file, print_show=self.print_show, r=self.r,noise_mode=self.noise_mode)                      
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            self.print_show = False
            # never show noisy rate again
            return trainloader, all_dataset.train_noisy_labels

        elif mode == 'train':
            labeled_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type,
                                             noise_path=self.noise_path, is_human=self.is_human,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="all_lab",
                                             noise_file=self.noise_file, pred=pred, probability=prob,probability2=prob2, log=self.log,
                                             transform_s=self.transform_train_s, r=self.r,noise_mode=self.noise_mode)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

            return labeled_trainloader, labeled_dataset.train_noisy_labels

        elif mode == 'test':
            test_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          is_human=self.is_human,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='test', r=self.r,noise_mode=self.noise_mode)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          is_human=self.is_human,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                          noise_file=self.noise_file, r=self.r,noise_mode=self.noise_mode)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader, eval_dataset.noise_or_not
        # never print again
