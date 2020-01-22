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
from dataset import Willow, Stanford10, VOC2012,ucf101,hmdb
from torch.nn import DataParallel
import numpy as np
#from sync_batchnorm import DataParallelWithCallback as DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=101, help="training epoch", type=int)
parser.add_argument('--batch_size', default=32, help="batch_size", type=int)
parser.add_argument('--lambda_pixel', default=10.0, help="the coefficient of mse and bce", type=float)
parser.add_argument('--lambda_mse', default=0.01, help="the coefficient of mse", type=float)
parser.add_argument('--lambda_E1', default=5.0, help="the coefficient of E1", type=float)
parser.add_argument('--lambda_E2', default=1, help="the coefficient of E2", type=float)
parser.add_argument('--lambda_cls', default=5, help="the coefficient of cls", type=float)
parser.add_argument('--save_path', default='./G/', help="save_path", type=str)
parser.add_argument('--interval', default=10, help="the interval of video frames", type=int)
parser.add_argument('--start_epoch', default=-1, help="the start epoch", type=int)
parser.add_argument('--dataset', default='Stanford10', help="which dataset", type=str)
parser.add_argument('--start_ckpt', default='', help="start ckpt", type=str)
parser.add_argument('--z_dim', default=8, help="the dimension of z", type=int)

dataset_classes = {"Willow":Willow, "Stanford10":Stanford10, "VOC2012":VOC2012}
num_category = {"Willow":7, "Stanford10":10, "VOC2012":10}
initial_ckpt={"Willow":'zxalexnet_willow.ckpt', "Stanford10":'zxalexnet.ckpt', "VOC2012":'zxalexnet_voc.ckpt'}
net_ckpt={"Willow":'willow_1024_6674.ckpt', "Stanford10":'standford10_1024_8313.ckpt', "VOC2012":'voc_1024_6512.ckpt'}

args = parser.parse_args()
Datasetclass = dataset_classes[args.dataset]
num_classes = num_category[args.dataset]
batch_size=args.batch_size
epochs = args.epochs
lambda_pixel=args.lambda_pixel
lambda_mse=args.lambda_mse
lambda_E1=args.lambda_E1
lambda_E2=args.lambda_E2
lambda_cls=args.lambda_cls
save_path=args.save_path
interval=args.interval
start_epoch=args.start_epoch
start_ckpt=args.start_ckpt
z_dim=args.z_dim
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
    def __init__(self,num_class=num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_class),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x


def train(prenet,net, trainloader, epochs=100):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,verbose=True)
    prenet.eval()
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
            nn.Conv2d(3+z_dim, 64, kernel_size=11, stride=4, padding=2),
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, z_dim),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x



def train_gan(G,D,E,cls,dataloader, cls_dataloader,epochs=30,lambda_pixel=1,save_path = './G/',start_epoch=-1):
    criterion_G1 = nn.MSELoss().cuda()
    criterion_G2=nn.BCELoss().cuda()
    criterion_G_cls =nn.CrossEntropyLoss().cuda()
    #optimizer_G = torch.optim.SGD(G.parameters(), lr=1e-3, momentum=0.9)
    optimizer_G=torch.optim.Adam(G.parameters(), lr=1e-4)
    criterion_D = nn.BCELoss().cuda()
    #optimizer_D = torch.optim.SGD(D.parameters(), lr=1e-3, momentum=0.9)
    optimizer_D =torch.optim.Adam(D.parameters(), lr=1e-4)
    criterion_E=nn.L1Loss().cuda()
    #optimizer_E=torch.optim.SGD(E.parameters(), lr=1e-3, momentum=0.9)
    optimizer_E=torch.optim.Adam(E.parameters(), lr=1e-4)

    G.train()
    D.train()
    E.train()
    f=open('gan_log.txt','a')
    for e in range(start_epoch+1,epochs):
        tq = tqdm(dataloader, ncols=80, ascii=True)
        for i, batch in enumerate(tq):
            batch_images=batch['image']
            #print('1',batch_images.shape)
            z = Variable(torch.randn((batch_images.size(0),z_dim)))
            z_img = z.view(z.size(0), z.size(1),1, 1).expand(z.size(0), z.size(1),batch_images.size(2), batch_images.size(3))
            #print('2',z_img.shape)
            batch_images = torch.cat([batch_images, z_img], 1)
            #print('3',batch_images.shape)
            batch_features=batch['feature']

            batch_images=batch_images.cuda()
            batch_features=batch_features.view(batch_features.size(0),4096)
            batch_features=batch_features.cuda()
            z=z.cuda()
            #valid = Variable(Tensor(np.ones((batch_features.size(0), *patch))), requires_grad=False)
            #fake = Variable(Tensor(np.zeros((batch_features.size(0), *patch))), requires_grad=False)
            valid = Variable(torch.ones(batch_features.size(0),1)).cuda()
            fake = Variable(torch.zeros(batch_features.size(0),1)).cuda()

            # ------------------
            #  Train Generators
            # ------------------
            '''
            fake_f = G(batch_images)
            pred_fake = D(fake_f)
            loss_bce = criterion_G2(pred_fake, valid)
            pred_z=E(fake_f)
            loss_E=criterion_E(pred_z,z)
            optimizer_E.zero_grad()
            loss_E.backward()
            optimizer_E.step()
            '''

            # Total loss
            #loss_G = (loss_bce + lambda_pixel * loss_mse)/(lambda_pixel+1)
            fake_f = G(batch_images)
            loss_mse = criterion_G1(fake_f, batch_features)
            pred_fake = D(fake_f)
            loss_bce = criterion_G2(pred_fake, valid)
            pred_z = E(fake_f)
            loss_E1 = criterion_E(pred_z, z)
            loss_G=loss_bce+lambda_mse*loss_mse      #?????????????????????????????????????????????????????/
            #optimizer_G.zero_grad()
            #loss_G.backward()
            #optimizer_G.step()

            for i in cls_dataloader:
                i_imgs,i_labels=i['image'],i['label']
                break
            i_z=Variable(torch.randn((i_imgs.size(0),z_dim)))
            i_imgs=torch.cat([i_imgs,i_z.view(i_z.size(0), i_z.size(1), 1, 1).expand(i_z.size(0), i_z.size(1), i_imgs.size(2), i_imgs.size(3))],1)
            i_imgs=i_imgs.cuda()
            i_labels=i_labels.cuda()
            cls_f=G(i_imgs)
            pre_cls=cls(cls_f)
            i_z=i_z.cuda()
            loss_E2=criterion_E(E(cls_f), i_z)
            loss_G_cls=lambda_cls*criterion_G_cls(pre_cls,i_labels)+loss_G+lambda_E1*loss_E1+lambda_E2*loss_E2#??????????????????????????????????????????
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            loss_G_cls.backward()
            optimizer_G.step()
            optimizer_E.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Real loss
            pred_real = D(batch_features)
            loss_real = criterion_D(pred_real, valid)
            # Fake loss
            pred_fake = D(fake_f.detach())
            loss_fake = criterion_D(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            print(e,'/',epochs,'  loss_G',loss_G.item(),'  loss_D',loss_D.item(),'  loss_G_cls',loss_G_cls.item(),'  loss_E1',loss_E1.item())
        f.write(str(e) + '/' + str(epochs) + '  loss_G:' + str(loss_G.item()) + '  loss_D:' + str(loss_D.item())+'  loss_G_cls:'+str(loss_G_cls.item()))
        f.write('\n')

        torch.save(G.state_dict(), save_path+'G'+str(e)+'.ckpt')
    f.close()




if __name__ == '__main__':
    pretrained_dict = torch.load(initial_ckpt[args.dataset])
    tmp = torch.randn((64, z_dim, 11, 11))*0.01
    pretrained_dict['features.0.weight'] = torch.cat([pretrained_dict['features.0.weight'], tmp], 1)

    cls_dataset = Datasetclass("train")
    cls_dataloader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=True)
    prenet = myAlexNet()
    prenet.load_state_dict(torch.load(initial_ckpt[args.dataset]), strict=True)
    prenet = prenet.cuda()
    cls=Classifier(num_class=num_classes).cuda()
    cls.load_state_dict(torch.load(net_ckpt[args.dataset]), strict=True)
    cls=cls.cuda()
    #train(prenet, cls, cls_dataloader,100)


    G=Generator()
    if start_ckpt=='':
        G.load_state_dict(pretrained_dict, strict=True)
    else:
        G.load_state_dict(torch.load(start_ckpt), strict=True)
    G=G.cuda()
    D=Discriminator().cuda()
    E=Encoder().cuda()
    dataloader = DataLoader(ucf101(interval), batch_size=batch_size, shuffle=True)
    train_gan(G, D, E,cls,dataloader, cls_dataloader,epochs=epochs, lambda_pixel=lambda_pixel, save_path=save_path,start_epoch=start_epoch)











