import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
import time
import argparse
import ast
from dataset import Willow, Stanford10, VOC2012,ucf101
from torch.nn import DataParallel
import random
import numpy as np
#from sync_batchnorm import DataParallelWithCallback as DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=101, help="training epoch", type=int)
parser.add_argument('--batch_size', default=32, help="batch_size", type=int)
parser.add_argument('--lambda_pixel', default=10.0, help="the coefficient of mse and bce", type=float)
parser.add_argument('--save_path', default='./G/', help="save_path", type=str)
parser.add_argument('--interval', default=25, help="the interval of video frames", type=int)
parser.add_argument('--start_epoch', default=-1, help="the start epoch", type=int)
parser.add_argument('--dataset', default='Stanford10', help="which dataset", type=str)
parser.add_argument('--start_ckpt', default='zxalexnet.ckpt', help="start ckpt", type=str)
parser.add_argument('--K', default=3, help="the number of Generator", type=int)

dataset_classes = {"Willow":Willow, "Stanford10":Stanford10, "VOC2012":VOC2012}
num_category = {"Willow":7, "Stanford10":10, "VOC2012":10}


args = parser.parse_args()
Datasetclass = dataset_classes[args.dataset]
num_classes = num_category[args.dataset]
batch_size=args.batch_size
epochs = args.epochs
lambda_pixel=args.lambda_pixel
save_path=args.save_path
interval=args.interval
start_epoch=args.start_epoch
start_ckpt=args.start_ckpt
K=args.K

#---------------------------pretrain Classifier-----------------------------------------------------------------------------------------------------
class myAlexNet(nn.Module):
    def __init__(self):
        super(myAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096)
        )
    def forward(self, x):
        x = self.features(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class Classifier(nn.Module):
    def __init__(self,num_class=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.classifier(x)
        return x


def train(prenet,net, trainloader, epochs=100):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,verbose=True)
    net.train()
    for epoch in range(epochs):
        tq = tqdm(trainloader, ncols=80, ascii=True)
        start_time=time.time()
        correct = 0
        total = 0
        for i, batch in enumerate(tq):
            image, label = batch['image'], batch['label']
            image=image.cuda()
            label=label.cuda()
            pre=net(prenet(image))
            loss=criterion(pre,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pre, 1)
            total += image.size(0)
            correct += (predicted == label).sum().item()
        #scheduler.step(loss)
        print("epoch:", epoch + 1,'/',epochs,'loss:',loss.item(),'acc:',correct/total, "time:", time.time() - start_time)

#-----------------------------------------------------------------------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096)
        )
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x=self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

def train_gan(K,G,D,cls,dataset, cls_dataloader,epochs=30,lambda_pixel=1,save_path = './G/',start_epoch=-1):
    criterion_G1 = list()
    criterion_G2 = list()
    criterion_G_cls = list()
    optimizer_G = list()
    criterion_D = list()
    optimizer_D = list()
    all_indexes = list()
    for i in range(K):
        criterion_G1.append(nn.MSELoss().cuda())
        criterion_G2.append(nn.BCELoss().cuda())
        criterion_G_cls.append(nn.CrossEntropyLoss().cuda())
        optimizer_G.append(torch.optim.SGD(G[i].parameters(), lr=1e-3, momentum=0.9))
        criterion_D.append(nn.BCELoss().cuda())
        optimizer_D.append(torch.optim.SGD(D[i].parameters(), lr=1e-3, momentum=0.9))
        G[i].train()
        D[i].train()
        all_indexes.append(list())
    for i in range(len(dataset)):
        ran=random.randint(0,K-1)
        all_indexes[ran].append(i)
    f=open('gan_log.txt','a')
    for e in range(start_epoch+1,epochs):
        for kk in range(K):
            random.shuffle(all_indexes[kk])
            index=0
            while index+batch_size<len(all_indexes[kk]):
                batch_images=list()
                batch_features=list()
                for i in range(index,index+batch_size):
                    batch_images.append(dataset[all_indexes[kk][i]]['image'].numpy())
                    batch_features.append(dataset[all_indexes[kk][i]]['feature'].numpy())
                index+=batch_size
                batch_images=torch.from_numpy(np.array(batch_images))
                batch_features=torch.from_numpy(np.array(batch_features))
                batch_features = batch_features.view(batch_size, 4096)
                batch_images = batch_images.cuda()
                batch_features = batch_features.cuda()
                valid = Variable(torch.ones(batch_size)).cuda()
                fake = Variable(torch.zeros(batch_size)).cuda()
                # ------------------
                #  Train Generators
                # ------------------
                fake_f = G[kk](batch_images)
                pred_fake = D[kk](fake_f)
                loss_bce = criterion_G2[kk](pred_fake, valid)
                loss_mse = criterion_G1[kk](fake_f, batch_features)
                # Total loss
                loss_G = (loss_bce + lambda_pixel * loss_mse) / (lambda_pixel + 1)
                # optimizer_G.zero_grad()
                # loss_G.backward()
                # optimizer_G.step()

                for i in cls_dataloader:
                    i_imgs, i_labels = i['image'], i['label']
                    break
                i_imgs = i_imgs.cuda()
                i_labels = i_labels.cuda()
                pre_cls = cls(G[kk](i_imgs))
                loss_G_cls = 5.0 * criterion_G_cls[kk](pre_cls, i_labels) + loss_G
                optimizer_G[kk].zero_grad()
                loss_G_cls.backward()
                optimizer_G[kk].step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Real loss
                pred_real = D[kk](batch_features)
                loss_real = criterion_D[kk](pred_real, valid)
                # Fake loss
                pred_fake = D[kk](fake_f.detach())
                loss_fake = criterion_D[kk](pred_fake, fake)
                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)
                optimizer_D[kk].zero_grad()
                loss_D.backward()
                optimizer_D[kk].step()
                print(e, '/', epochs, '  loss_G', loss_G.item(), '  loss_D', loss_D.item(), '  loss_G_cls',loss_G_cls.item())
        f.write(str(e) + '/' + str(epochs) + '  loss_G:' + str(loss_G.item()) + '  loss_D:' + str(loss_D.item())+'  loss_G_cls:'+str(loss_G_cls.item()))
        f.write('\n')
        if e%10==0:
            for kk in range(K):
                torch.save(G[kk].state_dict(), save_path+'G'+str(e)+'_'+str(kk)+'.ckpt')
        # ---------------------
        #  get all_indexes
        # ---------------------
        for kk in range(K):
            all_indexes[kk].clear()
        for i in range(len(dataset)):
            image=dataset[i]['image'].cuda()
            feature=dataset[i]['feature'].cuda()
            image=image.view(1,3,224,224)
            index0=0
            cri_loss=1000.0
            for kk in range(K):
                fake_f = G[kk](image)
                pred_fake = D[kk](fake_f)
                loss_bce = criterion_G2[kk](pred_fake, Variable(torch.ones(1)).cuda())
                loss_mse = criterion_G1[kk](fake_f, feature)
                loss_G = (loss_bce + lambda_pixel * loss_mse) / (lambda_pixel + 1)
                if loss_G<cri_loss:
                    cri_loss=loss_G
                    index0=kk
            all_indexes[index0].append(i)
    f.close()




if __name__ == '__main__':
    cls_dataset = Datasetclass("train")
    cls_dataloader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=True)
    prenet = myAlexNet()
    prenet.load_state_dict(torch.load('zxalexnet.ckpt'), strict=True)
    prenet = prenet.cuda()
    cls=Classifier(num_class=num_classes).cuda()
    train(prenet, cls, cls_dataloader, 100)

    G=list()
    D=list()
    for i in range(K):
        G.append(Generator())
        G[i].load_state_dict(torch.load(start_ckpt), strict=True)
        G[i]=G[i].cuda()
        D.append(Discriminator().cuda())
    dataset=ucf101(interval)


    #G=Generator()
    #G.load_state_dict(torch.load(start_ckpt), strict=True)
    #G=G.cuda()
    #D=Discriminator().cuda()
    #dataloader = DataLoader(ucf101(interval), batch_size=batch_size, shuffle=True)
    train_gan(K,G, D, cls,dataset, cls_dataloader,epochs=epochs, lambda_pixel=lambda_pixel, save_path=save_path,start_epoch=start_epoch)













