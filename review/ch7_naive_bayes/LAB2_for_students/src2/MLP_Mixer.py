import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）

        # use sequential to def a mlp
        class MlpBlock(nn.Module):
            def __init__(self, in_dim):
                super(MlpBlock, self).__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(),
                    nn.Linear(hidden_dim, in_dim),
                    nn.Dropout())

            def forward(self, x):
                return self.mlp(x)

        patch_number = (28 // patch_size) ** 2  # 序列数
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.tokenMix = MlpBlock(in_dim=patch_number)
        self.channelMix = MlpBlock(in_dim=hidden_dim)
        ########################################################################

    def forward(self, x):
        ########################################################################
        # x = self.norm1(x)
        # x1 = x.transpose(1, 2)
        # x1 = self.linear1(x1)
        # x1 = self.gelu(x1)
        # x1 = self.dropout(x1)
        # x1 = self.linear1(x1)
        # x1 = self.dropout(x1)
        # x1 = x1.transpose(1, 2)
        # u = x + x1
        # u1 = u.transpose(1, 2)
        # u1 = self.linear1(u1)
        # u1 = self.gelu(u1)
        # u1 = self.dropout(u1)
        # u1 = self.linear1(u1)
        # u1 = self.dropout(u1)
        # u1 = u1.transpose(1, 2)
        # y = u + u1
        # return y
        x = self.norm1(x)
        x1 = self.tokenMix(x.transpose(1, 2)).transpose(1, 2)  # transport
        u = x + x1  # U = X + f_{MLP1}(X)
        u = self.norm2(u)
        u1 = self.channelMix(u)
        y = u + u1
        return y  # Y = U + f_{MLP2}(u)

        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        patch_number = (28 // patch_size) ** 2
        # pre-patch
        self.patch_embed = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        MixBlock = Mixer_Layer(patch_size, hidden_dim)
        self.mixlayers = nn.Sequential(*[MixBlock for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_last = nn.Linear(hidden_dim, 10)
    ########################################################################

    def forward(self, data):
        ########################################################################
        # 注意维度的变化
        x = self.patch_embed(data).flatten(2).transpose(1, 2)
        x = self.mixlayers(x)
        x = self.norm(x)
        x = torch.mean(x, dim=1)  # average pooling
        x = self.fc_last(x)
        return x
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    sum_loss = 0
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 计算loss并进行优化
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 需要计算测试集的loss和accuracy
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss
            _, pred = torch.max(output.data, 1)
            num_correct += torch.sum(pred == target)
        accuracy = num_correct / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    model = MLPMixer(patch_size=7, hidden_dim=128, depth=3).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
