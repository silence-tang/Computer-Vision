import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 定义超参数如下：
# 学习率
lr = 0.0005
momentum = 0.9
# 遍历训练集10次
epochs = 10
# 每次训练的批大小为64张图片   注：训练集共60000张图片
train_batch_size = 64
# 每次测试的批大小为128张图片   注：测试集共10000张图片
test_batch_size = 128


# 定义单个稠密块中的卷积层构造函数
def conv_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )


# 定义单个稠密块的构造函数
class DenseBlock(nn.Module):
    def __init__(self, convs_num, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        dense_blk = []
        for i in range(convs_num):
            # 保证每个卷积层的输出通道数相等
            temp_conv = conv_blk(in_channels=i * out_channels + in_channels, out_channels=out_channels)
            dense_blk.append(temp_conv)
        self.dense_block = nn.Sequential(*dense_blk)
        
    # 定义稠密块的前向计算方式
    def forward(self, x):
        for layer in self.dense_block:
            y = layer(x)
            # 将该层前面所有卷积层的输出在通道维度上连接，作为该层的输入
            x = torch.cat((x, y), dim=1)
        return x


# 定义过渡层，减小通道数和图像尺寸
# 在实际应用时，可认为out_channels = in_channels / 2，即将输入通道数减小一半
# 经过过渡层，图像尺寸也会减小一半
class trans_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(trans_block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    # 定义过渡块的前向计算方式
    def forward(self, x):
        x = self.conv(x)
        return x

# densenet类
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        # 一开始的卷积层，输出大小：24x24
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 构造4个稠密块，每个块中有4个卷积层，每两个块之间有一个过渡块。输出大小：3x3
        in_channels = 64           # in_channels一开始是第一个稠密块的输入通道数
        out_channels = 32          # out_channels是单个稠密块中每个卷积层的输出通道数
        convs_num = [4, 4, 4, 4]   # conv_num是每个稠密块中卷积层的数量
        dense_blks = []
        for i, num in enumerate(convs_num):
            dense_blks.append(DenseBlock(num, in_channels, out_channels))
            # 下一个稠密块的输入通道数
            in_channels += num * out_channels
            # 在每个稠密块之后添加转换层（最后一块除外），使输出通道数减半
            if i != len(convs_num) - 1:
                # 整数除法，必须用//
                dense_blks.append(trans_block(in_channels, in_channels // 2))
                # 通过过渡层之后，下一稠密块的输入通道数已经减半
                in_channels = in_channels // 2
        self.denseblocks = nn.Sequential(
            *dense_blks
        )
        # 最后的全剧平均池化+全连接输出层。输出大小：1x10
        self.final = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 10)
        )
    # 定义整个网络的前向计算方式
    def forward(self, x):
        x = self.conv(x)
        x = self.denseblocks(x)
        x = self.final(x)
        return x

# 实例化googlenet对象
densenet = DenseNet()
# 将网络移至GPU上训练
densenet = densenet.cuda()
# 初始化优化器
optimizer = optim.Adam(densenet.parameters(), lr = lr)
# 采用softmax+交叉熵函数计算损失
criterion = nn.CrossEntropyLoss()


# 初始化存放训练误差的列表，用于保存每批训练误差
loss_list = []
trained_count = []
# 定义训练过程
def train(epoch):
    # 设置为训练模式，启用drop out
    densenet.train()
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
        output = densenet(input)
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
    densenet.eval()
    # 初始化测试损失值为0
    test_loss = 0  
    # 初始化预测正确的数据个数为0
    correct = 0  
    for data, target in test_loader:
        # 将输入和目标值包装为Variable并迁移到GPU上
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # 计算输出
        with torch.no_grad():
            output = densenet(data)
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
    state = {'net':densenet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(densenet, r'C:\Users\HP\Desktop\PythonLearning\model\DenseNet_Trained_Model\DenseNet.pkl')