from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader

class Willow(Dataset):
    def __init__(self, mode='train'):
        if mode=='train':
            self.txtpath = 'datasets/Willow/trainval.txt'
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        elif mode=='test':
            self.txtpath = 'datasets/Willow/test.txt'
            self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        f = open(self.txtpath)
        self.data = f.readlines()
        self.JPEGroot = 'datasets/Willow/JPEGImages/'
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx].split()
        img = self.JPEGroot + img + '.jpg'
        label = int(label)
        img = Image.open(img)
        sample = {'image': self.transform(img), 'label': label}
        return sample

class Stanford10(Dataset):
    def __init__(self, mode='train'):
        self.category_num = {'climbing':0, 'fishing':1, 'jumping':2, 'playing_guitar':3, 'riding_a_bike':4,
            'riding_a_horse':5, 'rowing_a_boat':6, 'running':7, 'throwing_frisby':8, 'walking_the_dog':9}
        self.mode = mode
        if mode=='train':
            self.txtpath = 'datasets/Stanford10/train.txt'
            self.transform = transforms.Compose([
                transforms.Resize([260,260]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.transform2 = transforms.Compose([
                transforms.Resize([260,260]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.JPEGroot = 'datasets/Stanford10/train/'
        elif mode=='test':
            self.txtpath = 'datasets/Stanford10/test.txt'
            self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.JPEGroot = 'datasets/Stanford10/test/'
        f = open(self.txtpath)
        self.data = f.readlines()
        f.close()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imgname = self.data[idx].strip()
        category = '_'.join(imgname.split('_')[:-1])
        imgpath = self.JPEGroot + category + '/' + imgname
        # print(imgpath)
        img = Image.open(imgpath)
        label = self.category_num[category]
        if self.mode == 'test':# or (img.size[0] >= 224 and img.size[1] >= 224):
            image=self.transform(img)
        else:
            image=self.transform2(img)
        sample = { 'image': image, 'label': label }
        return sample



class VOC2012(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.txtpath = 'datasets/VOC2012/train.txt'
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.transform2 = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.JPEGroot = 'datasets/VOC2012/train/'
        elif mode=='test':
            self.txtpath = 'datasets/VOC2012/test.txt'
            self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
            self.JPEGroot = 'datasets/VOC2012/test/'
        f = open(self.txtpath)
        self.data = f.readlines()
        f.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx].split()
        img = self.JPEGroot + img
        label = int(label)
        img = Image.open(img)
        if self.mode == 'test' or (img.size[0] >= 224 and img.size[1] >= 224):
            image = self.transform(img)
        else:
            image = self.transform2(img)
        sample = {'image': image, 'label': label}
        return sample

class ucf101(Dataset):
    def __init__(self,interval):
        self.imgtxtpath='./datasets/ucf_imgs'+str(interval)+'.txt'
        self.ftxtpath='./datasets/ucf_features'+str(interval)+'.txt'
        self.transform = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform2 = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.root='./datasets'
        f_feature = open(self.root+'/ucf_features'+str(interval)+'.txt')
        f_imgs = open(self.root+'/ucf_imgs'+str(interval)+'.txt')
        self.f_features_list=f_feature.readlines()
        self.f_imgs_list=f_imgs.readlines()
        f_feature.close()
        f_imgs.close()

    def __len__(self):
        return len(self.f_features_list)

    def __getitem__(self, idx):
        imgpath=self.root+self.f_imgs_list[idx].strip()
        fpath=self.root+self.f_features_list[idx].strip()
        img = Image.open(imgpath)
        f=np.load(fpath)
        f=torch.from_numpy(f)

        if (img.size[0] >= 224 and img.size[1] >= 224):
            image=self.transform(img)
        else:
            image=self.transform2(img)
        sample = { 'image': image, 'feature': f }
        return sample


class hmdb(Dataset):
    def __init__(self,interval):
        self.imgtxtpath='./datasets/hmdb_imgs'+str(interval)+'.txt'
        self.ftxtpath='./datasets/hmdb_features'+str(interval)+'.txt'
        self.transform = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform2 = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.root='./datasets'
        f_feature = open(self.root+'/hmdb_features'+str(interval)+'.txt')
        f_imgs = open(self.root+'/hmdb_imgs'+str(interval)+'.txt')
        self.f_features_list=f_feature.readlines()
        self.f_imgs_list=f_imgs.readlines()
        f_feature.close()
        f_imgs.close()

    def __len__(self):
        return len(self.f_features_list)

    def __getitem__(self, idx):
        imgpath=self.root+self.f_imgs_list[idx].strip()
        fpath=self.root+self.f_features_list[idx].strip()
        img = Image.open(imgpath)
        f=np.load(fpath)
        f=torch.from_numpy(f)

        if (img.size[0] >= 224 and img.size[1] >= 224):
            image=self.transform(img)
        else:
            image=self.transform2(img)
        sample = { 'image': image, 'feature': f }
        return sample


if __name__ == "__main__":
    # testWillow = Willow('test')
    # print(len(testWillow))
    # print(testWillow[425])
    # print(type(testWillow[425]))
    # print(testWillow[425][0].shape)
    # print(testWillow[425][1])


    data=ucf101(10)
    print(data[0:1])

    # print(testS[00])
    # print(type(testS[1671]))
    # for i in range(len(testS)):
    #     print(testS[i][0].shape)
    #     print(testS[i][1])

    # testV = VOC2012('test')
    # print(len(testV))
    # print(testV[00])
    # print(type(testV[1671]))
    # for i in range(len(testV)):
    #     print(testV[i][0].shape)
    #     print(testV[i][1])

