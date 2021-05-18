import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import numpy as np
import os

DATA_PATH = '/data/wangruifeng/datasets/GANs/widget9k'
PARAMS_PATH = '/data/wangruifeng/params/GANs/widget9k/'
MODLE_PATH = '/data/wangruifeng/models/GANs/widget9k/'
#https://blog.csdn.net/qq_41380292/article/details/108686567?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242
# 首先自定义数据集吧：
from torch.utils.data import Dataset,DataLoader

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]
    '''
    首先说明一下以上的初始化参数，filepath是数据集的路径，transform是对源数据（features）的一些变化，target_transform是对目标数据（labels）的一些变换，keys是键，因为我的数据是这样的，整体是字典格式的，每个键对应的值又是ndarray数据，所以我通过键来索引对应的值 
    '''
    def __getitem__(self,mask):
        return self.data[mask], self.label[mask]
    def __len__(self):
        return self.length


# 然后再继续改动


'''

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
if not os.path.exists(DATA_PATH+'cnn-img'):
    os.mkdir(DATA_PATH+'cnn-img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

#z_dimension = 100


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #transforms.Normalize(mean=0.5, std=0.5)
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST dataset
mnist = datasets.MNIST(
    root=DATA_PATH+'data/', train=True, transform=img_transform, download=True)
# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)
'''


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Y_train为空,外界提供
def runTrain(X_train,Y_train=[]):
    num_epochs = 100
    batch_size = 128
    #batch_size = 64
    learning_rate = 1e-3
    if Y_train == []:
        Y_train = np.zeros(X_train.shape[0])
    train_set = MyDataSet(data=X_train, label=Y_train)
    dataloader = torch.utils.data.DataLoader(
       dataset=train_set, batch_size=batch_size, shuffle=True)

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, DATA_PATH+'/fake_images-{}.png'.format(epoch))

    torch.save(model.state_dict(), MODLE_PATH+'./widget_autoencoder.pth')
