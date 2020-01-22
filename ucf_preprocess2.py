import os
import imageio
import numpy as np
import skimage
from PIL import Image
import torch
from torch import nn
from torchvision import models
from torchvision import transforms

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

layer1=myAlexNet()
layer1.load_state_dict(torch.load('zxalexnet.ckpt'),strict=True)


tran=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])







interval=10


path='./UCF-101/'
file_names = os.listdir(path)
print(file_names)
f_feature=open('ucf_features'+str(interval)+'.txt','w')
f_imgs=open('ucf_imgs'+str(interval)+'.txt','w')
num=0
for file_name in file_names:
    new_path=path+file_name+'/'
    for v_name in os.listdir(new_path):
        print(num)
        num+=1
        v_path=new_path+v_name
        video= imageio.get_reader(v_path)
        for im in enumerate(video):
            if im[0]==5:
                img5=Image.fromarray(np.uint8(im[1]))
            if im[0]==(5+interval):
                img30=Image.fromarray(np.uint8(im[1]))
                feature30=layer1(tran(img30).unsqueeze(0))
                feature30=feature30.detach().numpy()
                feature30.squeeze(0)
                img5.save('./ucf_imgs'+str(interval)+'/a'+str(num)+'.jpg')
                f_imgs.write('/ucf_imgs'+str(interval)+'/a'+str(num)+'.jpg\n')
                np.save('./ucf_f'+str(interval)+'/a'+str(num)+'.npy',feature30)
                f_feature.write('/ucf_f'+str(interval)+'/a'+str(num)+'.npy\n')
            if im[0]==10:
                img10=Image.fromarray(np.uint8(im[1]))
            if im[0]==(10+interval):
                img35=Image.fromarray(np.uint8(im[1]))
                feature35=layer1(tran(img35).unsqueeze(0))
                feature35=feature35.detach().numpy()
                feature35.squeeze(0)
                img10.save('./ucf_imgs'+str(interval)+'/b'+str(num)+'.jpg')
                f_imgs.write('/ucf_imgs'+str(interval)+'/b'+str(num)+'.jpg\n')
                np.save('./ucf_f'+str(interval)+'/b'+str(num)+'.npy',feature35)
                f_feature.write('/ucf_f'+str(interval)+'/b'+str(num)+'.npy\n')
            if im[0]==15:
                img15=Image.fromarray(np.uint8(im[1]))
            if im[0]==(15+interval):
                img40=Image.fromarray(np.uint8(im[1]))
                feature40=layer1(tran(img40).unsqueeze(0))
                feature40=feature40.detach().numpy()
                feature40.squeeze(0)
                img15.save('./ucf_imgs'+str(interval)+'/c'+str(num)+'.jpg')
                f_imgs.write('/ucf_imgs'+str(interval)+'/c'+str(num)+'.jpg\n')
                np.save('./ucf_f'+str(interval)+'/c'+str(num)+'.npy',feature40)
                f_feature.write('/ucf_f'+str(interval)+'/c'+str(num)+'.npy\n')
                break


f_feature.close()
f_imgs.close()
#torch.numpy() torch.from_numpy()

