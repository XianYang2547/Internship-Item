import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from arc_loss import Arc_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2, 10)
        )
        self.arc = Arc_loss(2, 10)
        self.nllloss = nn.NLLLoss()

    def forward(self, x, labels):
        x = x.reshape(-1, 28 * 28)
        feature = self.fc1(x)
        outputs = self.fc2(feature)

        loss_arc = self.arc(feature)
        loss = self.nllloss(loss_arc, labels)
        # loss = loss_cls + loss_arc

        return feature, outputs, loss


# 可视化特征数据
def visualize(feat, labels, epoch):
    plt.ion()
    c = ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff",
         "#ff00ff", "#990000", "#999900", "#009900", "#009999"]
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], ".", c=c[i])
    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
    plt.title("epoch=%d" % epoch)
    plt.savefig("data_and_result/result.jpg")
    plt.draw()
    plt.pause(0.01)


train_data = torchvision.datasets.MNIST("data_and_result", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("data_and_result", train=False, transform=transforms.ToTensor(), download=True)
# 利用DataLoader来加载数据集
train_dataloader = data.DataLoader(train_data, batch_size=512, shuffle=True)
test_dataloader = data.DataLoader(test_data, batch_size=512, shuffle=True)

if __name__ == '__main__':
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    epoch = 0
    while True:
        net.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            feature, out, loss = net.forward(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(loss.item())
                # print(net.center_loss_layer.center)
        print('#-------test--------#')
        with torch.no_grad():
            net.eval()
            feat_loader = []
            label_loader = []
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                feature, out, _ = net(x, y)
                feat_loader.append(feature)
                label_loader.append((y))
            feat = torch.cat(feat_loader, 0)
            labels = torch.cat(label_loader, 0)
            visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch += 1
