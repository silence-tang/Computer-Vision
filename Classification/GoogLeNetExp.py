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
lr = 0.0005
momentum = 0.9
# 遍历训练集10次
epochs = 10
# 每次训练的批大小为128张图片   注：训练集共60000张图片
train_batch_size = 64
# 每次测试的批大小为128张图片   注：测试集共10000张图片
test_batch_size = 128


# 单个Inception块
class Inception_block(nn.Module):
    # c1--c4是每条路径的输出通道数
    # 构造初始化方法
    # c1 c4是int，c2 c3是tuple。c2[0] c2[1]分别是路径2第一个、第二个卷积层的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception_block, self).__init__()
        # 路径1，只有一个1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1,    kernel_size=1, padding=0)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1, padding=0)
        self.p2_2 = nn.Conv2d(in_channels=c2[0],       out_channels=c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0],       out_channels=c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 池化层不改变通道数
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        # 计算4个路径各自的输出
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)  # dim=0是批次大小，dim=1是通道数，后面两维是高宽


# GoogLeNet类
class GoogLeNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(GoogLeNet, self).__init__()
        # conv1: 1卷积+1池化，输入图像大小为96，输出图像大小为24x24
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2: 2卷积+1池化，输入图像大小为96，输出图像大小为12x12
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # inception1: 2 inception块 + 池化，输出图像大小为6x6
        self.inception1 = nn.Sequential(
            Inception_block(192, 64, (96, 128), (16, 32), 32),     # 第1个Inception块的输出通道数为64+128+32+32=256
            Inception_block(256, 128, (128, 192), (32, 96), 64),   # 第2个Inception块的输出通道数为128+192+96+64=480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # inception2: 5 inception块 + 池化，输出图像大小为3x3
        self.inception2 = nn.Sequential(
            Inception_block(480, 192, (96, 208), (16, 48), 64),    # 第1个Inception块的输出通道数为192+208+48+64=512
            Inception_block(512, 160, (112, 224), (24, 64), 64),    # 第2个Inception块的输出通道数为160+224+64+64=512
            Inception_block(512, 128, (128, 256), (24, 64), 64),   # 第3个Inception块的输出通道数为128+256+64+64=512
            Inception_block(512, 112, (144, 288), (32, 64), 64),   # 第4个Inception块的输出通道数为112+288+64+64=528
            Inception_block(528, 256, (160, 320), (32, 128), 128), # 第5个Inception块的输出通道数为256+320+128+128=832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # inception3: 2 inception块 + 池化，输出图像大小为3x3
        self.inception3 = nn.Sequential(
            Inception_block(832, 256, (160, 320), (32, 128), 128), # 第1个Inception块的输出通道数为256+320+128+128=832
            Inception_block(832, 384, (192, 384), (48, 128), 128), # 第2个Inception块的输出通道数为384+384+128+128=1024
        )
        # final: 1全局平均汇聚+1全连接，输出图像大小为1x1
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # Avgpool输出通道数为1024，大小为1x1(即1024x1x1)
            nn.Flatten(),                 # 展平为1x1024
            nn.Linear(1024, classes)      # 最终输出尺寸为1x10
        )
    # 网络的前馈计算
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.final(x)
        return x

# 实例化googlenet对象
googlenet = GoogLeNet(1, 10)
# 将网络移至GPU上训练
googlenet = googlenet.cuda()
# 初始化优化器
optimizer = optim.Adam(googlenet.parameters(), lr = lr)
# 采用softmax+交叉熵函数计算损失
criterion = nn.CrossEntropyLoss()


# 初始化存放训练误差的列表，用于保存每批训练误差
loss_list = []
trained_count = []
# 定义训练过程
def train(epoch):
    # 设置为训练模式，启用drop out
    googlenet.train()
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
        output = googlenet(input)
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
    googlenet.eval()
    # 初始化测试损失值为0
    test_loss = 0  
    # 初始化预测正确的数据个数为0
    correct = 0  
    for data, target in test_loader:
        # 将输入和目标值包装为Variable并迁移到GPU上
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # 计算输出
        with torch.no_grad():
            output = googlenet(data)
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
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train = True, download = False, transform = transforms.Compose([transforms.ToTensor(),  transforms.Resize((224, 224))]))
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train = False, download = False, transform = transforms.Compose([transforms.ToTensor(),  transforms.Resize((224, 224))])) 
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
    state = {'net':googlenet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(googlenet, r'C:\Users\HP\Desktop\PythonLearning\model\GoogLeNet_Trained_Model\GoogLeNet.pkl')