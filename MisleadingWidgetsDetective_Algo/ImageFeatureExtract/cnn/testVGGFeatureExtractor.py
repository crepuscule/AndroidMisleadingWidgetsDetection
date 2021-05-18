import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

pthfile = '/data/wangruifeng/params/vgg16-397923af.pth'

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg16(pretrained=True)
        #
        VGG.load_state_dict(torch.load(pthfile))
        #
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
 
 
model = Encoder()
model = model.cuda()
 
 
def extractor(img_path, net, use_gpu):
#def extractor(img_path, saved_path, net, use_gpu):
#transforms.Resize(256),
#transforms.CenterCrop(224),
    transform = transforms.Compose([
        transforms.ToTensor()]
    )
 
    #img = Image.open(img_path).convert('RGB')
    # 传入灰度图
    if isinstance(img_path,str):
        img = Image.open(img_path).convert('L').convert('RGB')
    else:
        img = img_path.convert('L').convert('RGB')
    img = transform(img)
 
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
 
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    print(y.shape)
    #np.savetxt(saved_path, y, delimiter=',')
    return y
 
 
def runExtract(picture_List):
    #extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
    '''
    files_list = []
    x = os.walk(data_dir)
    for path,d,filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))
 
    print(files_list)
    '''
    files_list = picture_List
 
    use_gpu = torch.cuda.is_available()
    #use_gpu = False
 
    featureList = []
    for x_path in files_list:
        featureList.append(extractor(x_path, model, use_gpu))
    return featureList
