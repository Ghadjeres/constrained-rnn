import torch
import torch.nn as nn
import torch.nn.functional as F
# contains all optimizers
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.FloatTensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Max pooling over a (2, 2) window
        # When stride is None: equals specified size!
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x)) # ~ reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test_simple():
    a = torch.ones(2, 3)

    # CUDA
    # if torch.cuda.is_available():
    #     a = a.cuda()
    #     print(a)

    x = Variable(torch.ones(2, 2), requires_grad=True)
    print(x)
    y = torch.add(x, 2)

    y = x + 2

    # applied on gradient when backward is called
    handle = y.register_hook(print)  # Can also return a new gradient?!
    # print(type(handle))
    # handle.remove()
    # y = x + 2
    print(type(y), y)

    z = y * y * 3
    z.register_hook(print)
    out = z.mean()
    print(z, out)

    print(x.grad)
    # out.backward()
    z.backward(torch.Tensor([[1.0, 0.1], [1.0, 0.1]]))
    # impossible!
    z.backward(torch.Tensor([[1.0, 0.1], [1.0, 0.1]]))

    print(x.grad)


if __name__ == '__main__':
    # test_simple()

    ### MEMO
    # input.unsqueeze(0) to add a fake batch dimension

    net = Net()
    print(net)
    # put network on gpu
    # net.cuda()

    # parameters is a generator
    print(list(net.parameters()))

    # create an optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # input and target must go on gpu too
    input = Variable(torch.rand(16, 1, 32, 32))
    output = net(input)
    target = Variable(torch.rand(16, 10))  # a dummy target, for example
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    # You need to clear the existing gradients though, else gradients will be accumulated to existing gradients
    net.zero_grad()
    loss.backward()

    optimizer.step()

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    print(trainloader)
