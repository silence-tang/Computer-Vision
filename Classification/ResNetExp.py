import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 定义超参数如下：
# 学习率
lr = 0.001
momentum = 0.9
# 遍历训练集10次
epochs = 10
# 每次训练的批大小为128张图片   注：训练集共60000张图片
train_batch_size = 128
# 每次测试的批大小为128张图片   注：测试集共10000张图片
test_batch_size = 128


# 定义单个残差块
class residual(nn.Module):
    def __init__(self, in_channels, out_channels, use1x1conv=False, stride=1):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use1x1conv == True:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.conv3 = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)

# 定义残差块构成的大块
def res_block(input_channels, num_channels, num, first_block=False): # first_block：是否是第一个残差块
    res_block = []
    for i in range(num):
        if i == 0 and not first_block:
            # 如果是大块中的第一个小残差块且该大块不是整个resnet中的第一个块，则需要1x1卷积
            res_block.append(residual(input_channels, num_channels, use1x1conv=True, stride=2))
        else:
            # 否则不需要1x1卷积
            res_block.append(residual(num_channels, num_channels))
    return res_block

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 先是一个卷积+池化，输出图像大小为24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第1个大的残差块组合（包含2个残差块），输出图像大小为24x24
        self.res_block1 = nn.Sequential(
            *res_block(64, 64, 2, first_block=True)
        )
        # 第2个大的残差块组合（包含2个残差块），输出图像大小为12x12
        self.res_block2 = nn.Sequential(
            *res_block(64, 128, 2, first_block=False)
        )
        # 第3个大的残差块组合（包含2个残差块），输出图像大小为6x6
        self.res_block3 = nn.Sequential(
            *res_block(128, 256, 2, first_block=False)
        )
        # 第4个大的残差块组合（包含2个残差块），输出图像大小为3x3
        self.res_block4 = nn.Sequential(
            *res_block(256, 512, 2, first_block=False)
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # 输出为512x1x1
            nn.Flatten(),                 # 输出为1x512
            nn.Linear(512, 10)            # 输出为1x10
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.final(x)
        return x


# 实例化googlenet对象
resnet = ResNet()
# 将网络移至GPU上训练
resnet = resnet.cuda()
# 初始化优化器
optimizer = optim.Adam(resnet.parameters(), lr = lr)
# 采用softmax+交叉熵函数计算损失
criterion = nn.CrossEntropyLoss()


# 初始化存放训练误差的列表，用于保存每批训练误差
loss_list = []
trained_count = []
# 定义训练过程
def train(epoch):
    # 设置为训练模式，启用drop out
    resnet.train()
    # 初始化存放训练误差的列表，用于保存各批训练结束后的误差
    loss_container = []
    trained_num = []
    # 遍历训练集中的每个batch，i是batch的下标，即第i个batch，实际上i和data是enumerate的两个分量
    for i, data in enumerate(train_loader):
        # data包含两个量：图像及其标签
        input, label = data
        # 将输入和目标值包装为Variable并迁移到GPU上
        input, label = Variable(input.cuda()), Variable(label.cuda())
        # 在计算之前先把梯度buffer清零
        optimizer.zero_grad()
        # 前向传播 + 计算误差 + 反向传播误差 + 更新参数
        # 前向传播，计算当前图像的预测值output
        output = resnet(input)
        # 计算真实值与预测值之间的误差
        loss = criterion(output, label)
        # 记录当前误差
        loss_container.append(loss.item())
        trained_num.append(i * train_batch_size)
        # 反向传播误差
        loss.backward()
        # 更新各矩阵参数
        optimizer.step()
        # 每训练50个batch输出训练信息（当前epoch、已训练图像个数、训练完成百分比、当前训练误差）
        if i % 50 == 0:
            print('训练集迭代次数：{}    ( {} / {} )    当前误差：{:.9f}'.format(epoch, i * train_batch_size, len(train_loader.dataset), loss.item()))
    # 将该次迭代的误差列表传入loss_list
    loss_list.append(loss_container)
    trained_count.append(trained_num)


def test():
    # 设置为测试模式，关闭drop out
    resnet.eval()
    # 初始化测试损失值为0
    test_loss = 0  
    # 初始化预测正确的数据个数为0
    correct = 0  
    for data, target in test_loader:
        # 将输入和目标值包装为Variable并迁移到GPU上
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # 计算输出
        with torch.no_grad():
            output = resnet(data)
            # 计算累计误差
            test_loss += criterion(output, target).item()
            # 计算预测类别的下标
            # torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index（也即预测的类别）
            pred = output.data.max(1, keepdim = True)[1]  #这里max(第一个参数为1表示取这一行的最大值)
            # 统计预测正确数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # 计算平均误差
    avg_loss = test_loss / len(test_loader.dataset)
    # 输出测试信息（平均误差、预测准确的图像个数、预测准确率）
    print('\n测试集平均误差：{:.9f}，准确率: {}/{}({:.4f}%)\n'.format(avg_loss, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    # 给出训练集和测试集，原始图像为28×28，为了适应AlexNet框架，我们在读入数据集时要resize一下
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train = True, download = False, transform = transforms.Compose([transforms.ToTensor(),  transforms.Resize((96, 96))]))
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train = False, download = False, transform = transforms.Compose([transforms.ToTensor(),  transforms.Resize((96, 96))])) 
    # 定义训练集、测试集数据的装载器
    train_loader = DataLoader(train_dataset ,batch_size = train_batch_size, shuffle = True)
    test_loader  = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False)
    # 开始训练模型及测试模型
    # 开始计时
    tic = time.time()
    for epoch in range(1, epochs + 1):
        # 先训练模型
        train(epoch)
        # 再测试模型
        test()
    # 结束计时
    toc = time.time()
    print('总耗时：%.4f秒' % float(toc - tic))
    # 打印训练误差列表
    #print(loss_list[0])
    #print(trained_count[0])
    # 绘制第一次epoch的训练误差下降曲线
    fig = plt.figure()
    plt.plot(trained_count[0], loss_list[0], color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('Number of trained examples')
    plt.ylabel('CrossEntropyLoss')
    plt.show()
    # 最后保存一下模型
    state = {'net':resnet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(resnet, r'C:\Users\HP\Desktop\PythonLearning\model\ResNet_Trained_Model\ResNet.pkl')