import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# 定义超参数如下：
# 学习率
lr = 0.01
# 动量因子
momentum = 0.9
# 遍历训练集10次
epochs = 10
# 每次训练的批大小为64张图片   注：训练集共60000张图片
train_batch_size = 64
# 每次测试的批大小为1024张图片 注：测试集共10000张图片
test_batch_size = 1024


class LeNet(nn.Module):
    def __init__(self):
        # 继承超类所有方法
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 输入图像只有1个通道，尺寸为28×28，padding=2保持卷积后图像大小不变，用6个5×5卷积核进行卷积，输出图尺寸为28×28
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 第一个池化层输出为6×14×14
            # 论文中激活函数用的sigmoid
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            # 第二个卷积层，输入为6×14×14，无填充，输出为16×10×10
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            #nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第二个池化层输出为16×5×5
            # 论文中激活函数用的sigmoid
            nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            # 论文中该层其实不是FC，且没有激活函数，这里简化了
            #nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84),
            #nn.ReLU()
            # 论文中激活函数用的tanh
            nn.Tanh()
        )
        # 这里没用径向基函数
        self.fc3 = nn.Linear(in_features = 84, out_features = 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)

    # 使用num_flat_features函数计算张量x的总特征量. 比如x是4*2*2的张量，那么它的特征总量就是16
    def num_flat_features(self, x):
        # nn.Conv2d允许输入4维的Tensor：n个样本 x n个通道 x 高度 x 宽度
        # 这里为了计算总特征数，我们只取后三个维度各自的特征，所以是[1:]
        size = x.size()[1:] 
        # 初始化特征数为1
        num_features = 1
        # 将各维度特征数相乘。假设：通道数=3，高=10，宽=10
        # 则num_features = 3*10*10 = 300
        for s in size:
            num_features *= s
        return num_features

# 实例化LeNet网络
lenet = LeNet()
#print(lenet)
# 将网络移动到GPU上训练
lenet = lenet.cuda()
# 初始化优化器
optimizer = optim.SGD(lenet.parameters(), lr=lr, momentum=momentum)
# 采用交叉熵函数计算loss
criterion = nn.CrossEntropyLoss()

# 初始化存放训练误差的列表，用于保存每批训练误差
loss_list = []
trained_count = []
# 定义训练过程
def train(epoch):
    # 设置为训练模式，启用BN和drop out
    lenet.train()
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
        output = lenet.forward(input)
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
    # 设置为测试模式，关闭BN和drop out
    lenet.eval()
    # 初始化测试损失值为0
    test_loss = 0  
    # 初始化预测正确的数据个数为0
    correct = 0  
    for data, target in test_loader:
        # 将输入和目标值包装为Variable并迁移到GPU上
        data, target = Variable(data.cuda()), Variable(target.cuda())
        # 计算输出
        output = lenet.forward(data)
        # 计算累计误差
        test_loss += criterion(output, target).item()
        # 计算预测类别的下标
        pred = output.data.max(1, keepdim=True)[1]
        # 统计预测正确数
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # 计算平均误差
    avg_loss = test_loss / len(test_loader.dataset)
    # 输出测试信息（平均误差、预测准确的图像个数、预测准确率）
    print('\n测试集平均误差：{:.9f}，准确率: {}/{}({:.4f}%)\n'.format(avg_loss, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    # 给出训练集和测试集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False,
                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,download=False,
                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    # 定义训练集、测试集数据的装载器
    train_loader = DataLoader(train_dataset ,batch_size = train_batch_size, shuffle = True)
    test_loader  = DataLoader(test_dataset, batch_size = test_batch_size, shuffle = False)
    
    # 开始训练模型及测试模型
    # epoch依次为1,2
    for epoch in range(1, epochs + 1):
        # 先训练模型
        train(epoch)
        # 再测试模型
        test()
    # 打印训练误差列表
    #print(loss_list[0])
    #print(trained_count[0])
    fig = plt.figure()
    plt.plot(trained_count[0], loss_list[0], color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('Number of trained examples')
    plt.ylabel('CrossEntropyLoss')
    plt.show()
    # 最后保存一下模型
    state = {'net':lenet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(lenet, r'C:\Users\HP\Desktop\PythonLearning\model\LeNet.pkl')
    