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
from dataset import Willow, Stanford10, VOC2012
from torch.nn import DataParallel
#from sync_batchnorm import DataParallelWithCallback as DataParallel

dataset_classes = {"Willow":Willow, "Stanford10":Stanford10, "VOC2012":VOC2012}
num_category = {"Willow":7, "Stanford10":10, "VOC2012":10}
backbone_dict = {"resnet50":models.resnet50, "alexnet":models.alexnet}

parser = argparse.ArgumentParser()
# --total parameter--
parser.add_argument('--batch_size', default=64, help="batch_size", type=int)
parser.add_argument('--workers', default=32, help="workers", type=int)
parser.add_argument('--dataset', default='Stanford10', help="which dataset", type=str)
parser.add_argument('--backbone', default='alexnet', help="which network backbone", type=str)
parser.add_argument('--mode', default='traintest', help="train or test or both", type=str)
parser.add_argument('--ckpt-path', default="standford10_2.ckpt", help="path of pth file", type=str)
#parser.add_argument('--prenet_weights_path', default='./G/G0.ckpt', help="path of prenet weights", type=str)
# --training parameter--
parser.add_argument('--pretrained', default=True, help="whether model pretrained on the ImageNet", type=ast.literal_eval)
parser.add_argument('--prenet_ckpt', default='zxalexnet.ckpt', help="which ckpt we will load", type=str)
parser.add_argument('--epochs', default=80, help="training epoch", type=int)
parser.add_argument('--continue-train', default=False, help="whether to train on the previous model", type=ast.literal_eval)
parser.add_argument('--z_dim', default=8, help="the dimension of z", type=int)
parser.add_argument('--K', default=3, help="the number of z", type=int)


args = parser.parse_args()
# --total parameter--
#prenet_weights_path=args.prenet_weights_path
batch_size=args.batch_size
workers=args.workers
prenet_ckpt=args.prenet_ckpt
Datasetclass = dataset_classes[args.dataset]
num_classes = num_category[args.dataset]
backbone = backbone_dict[args.backbone]
mode = args.mode
ckpt_path = args.ckpt_path
z_dim=args.z_dim
K=args.K
# --training parameter--
whether_pretrained = args.pretrained
epochs = args.epochs
whether_continue = args.continue_train
print('dataset:',args.dataset)
print('backbone:',args.backbone)
print('mode:',args.mode)
print('ckpt-path:',args.ckpt_path)
print('prenet_ckpt:',prenet_ckpt)
if args.mode in ['traintest','train']:
    print('pretrained:',args.pretrained)
    print('epoch:',args.epochs)
    print('continue:',args.continue_train)

class myAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 自适应平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class myAlexNet2(nn.Module):
    def __init__(self, num_classes=1000):
        super(myAlexNet2, self).__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 自适应平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

'''

class model(nn.Module):
    def __init__(self, num_class=10):
        super(model,self).__init__()
        self.layer1=myAlexNet()
        alexnet = models.alexnet(pretrained=whether_pretrained)
        pretrained_dict = alexnet.state_dict()
        model_dict = self.layer1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.layer1.load_state_dict(model_dict)

        self.layer2 = nn.Linear(4096, 1024)
        self.layer3=nn.Dropout(p=0.5)
        self.layer4 = nn.Linear(1024, num_class)
        self.layer5=nn.Softmax()


    def forward(self,x):
        with torch.no_grad():
            x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.layer5(x)
        return x

class model0(nn.Module):
    def __init__(self, num_class=10):
        super(model,self).__init__()
        self.backbone = backbone(pretrained=whether_pretrained)
        self.linear = nn.Linear(1000, num_class)
    def forward(self,x):
        x = self.backbone(x)
        x = self.linear(x)
        return x
'''
class model(nn.Module):
    def __init__(self, num_class=10):
        super(model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096*2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_class),
            nn.Softmax(dim=1)
        )
        #self.layer1 = nn.Linear(4096, 4096)
        #nn.ReLU(inplace=True),
        #self.layer2 = nn.Dropout(0.5)
        #self.layer3 = nn.Linear(4096, num_class)
        #self.layer4=nn.Softmax()
    def forward(self, x):
        x = self.classifier(x)
        return x



def train(prenet1,prenet2,net, trainloader, epochs=30,save_path = 'stanford10.ckpt'):
    if whether_continue:
        net.load_state_dict(torch.load(save_path))
    f=open('train_log.txt','a')
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,verbose=True)
    prenet1.eval()
    prenet2.eval()
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
            z = Variable(torch.randn((image.size(0), z_dim)))
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
            z_img = z_img.cuda()
            f_f = prenet2(torch.cat([image, z_img], 1))
            for i in range(1, K):
                z = Variable(torch.randn((image.size(0), z_dim)))
                z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
                z_img = z_img.cuda()
                f_f += prenet2(torch.cat([image, z_img], 1))
            f_f /= K

            pre = net(torch.cat([prenet1(image), f_f], dim=1))
            loss=criterion(pre,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pre, 1)
            total += image.size(0)
            correct += (predicted == label).sum().item()
        #scheduler.step(loss)
        print("epoch:", epoch + 1,'/',epochs,'loss:',loss.item(),'acc:',correct/total, "time:", time.time() - start_time)
        f.write("epoch:"+str(epoch + 1)+'/'+str(epochs)+'  loss:'+str(loss.item())+'   acc:'+str(correct/total))
        f.write('\n')
    torch.save(net.state_dict(), save_path)
    f.close()

def test(prenet1,prenet2,net, testloader, load_path = 'stanford10.ckpt'):
    net.load_state_dict(torch.load(load_path))
    prenet1.eval()
    prenet2.eval()
    net.eval()
    tq = tqdm(testloader)
    start_time = time.time()

    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(tq):
            # label = label.cuda(async=True)
            image, label = batch['image'], batch['label']
            image = image.cuda()
            label = label.cuda()
            z = Variable(torch.randn((image.size(0), z_dim)))
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
            z_img = z_img.cuda()
            f_f = prenet2(torch.cat([image, z_img], 1))
            for i in range(1, K):
                z = Variable(torch.randn((image.size(0), z_dim)))
                z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
                z_img = z_img.cuda()
                f_f += prenet2(torch.cat([image, z_img], 1))
            f_f /= K

            pre = net(torch.cat([prenet1(image), f_f], dim=1))
            _, predicted = torch.max(pre, 1)
            # print('t o p', target_var.shape, output.shape, predicted.shape)
            total += image.size(0)
            correct += (predicted == label).sum().item()
        f = open('train_log.txt', 'a')
        f.write('acc: '+str(correct/total))
        f.write('\n')
        print('test accuracy: {:4f}'.format(correct/total))
        f.close()
    print('test time:', time.time()-start_time)


if __name__ == '__main__':
    trainset = Datasetclass("train")
    testset = Datasetclass("test")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    #trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=workers)
    #testloader = DataLoader(testset, batch_size=batch_size,num_workers=workers)

    prenet1=myAlexNet()
    prenet2 = myAlexNet2()
    '''
     alexnet = models.alexnet(pretrained=whether_pretrained)
    pretrained_dict = alexnet.state_dict()
    model_dict = prenet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    prenet.load_state_dict(model_dict)
    '''
    prenet1.load_state_dict(torch.load('zxalexnet.ckpt'),strict=True)
    prenet1=prenet1.cuda()
    prenet2.load_state_dict(torch.load(prenet_ckpt), strict=True)
    prenet2 = prenet2.cuda()
    net = model(num_class=num_classes).cuda()
    #net = DataParallel(net)
    f = open('train_log.txt', 'a')
    f.write(prenet_ckpt)
    f.write('\n')
    f.close()

    # print(net)
    if mode == "traintest":
        train(prenet1,prenet2,net, trainloader,epochs, ckpt_path)
        test(prenet1,prenet2,net, testloader, ckpt_path)
    elif mode == "train":
        train(prenet1,prenet2,net, trainloader, epochs,ckpt_path)
    else:
        test(prenet1,prenet2,net, testloader, ckpt_path)



