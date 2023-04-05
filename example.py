import argparse
from matplotlib import pyplot as plt

import torch
from torch import nn

import activation

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from logicGateDataset.datasets import AndGate, OrGate, XorGate, NotGate


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.layer1 = nn.Linear(2, 1)
        self.act1 = activation.Swift(slope=1.)

        # self.layer2 = nn.Linear(30, 10)
        # self.act2 = swift(slope=2.)

    def forward(self, input):
        out = self.act1(self.layer1(input))
        # out = self.act2(self.layer2(out))
        return out
    

def mnistDataLoader(train: bool, batch_size: int) -> DataLoader:
    '''
    pytorch MNIST train & test 데이터 로더 반환
    + z-normalization
    + Flatten
    '''
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.1307, ), std=(0.3081, )),
        Lambda(lambda x: torch.flatten(x)),
    ])

    loader = DataLoader(
        MNIST(
            root='./mnist/',
            train=train,
            transform=transform,
            download=True
        ), shuffle=train, batch_size=batch_size
    )
    return loader


if __name__ == '__main__':
    # dataset = XorGate(dataset_size=5000)
    # dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataLoader = mnistDataLoader(train=True, batch_size=64)

    model = Model()

    cri = nn.MSELoss()
    # cri = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = list()
    for x, y in dataLoader:
        y_hat = model(x)
        loss = cri(y_hat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss)
        losses.append(loss.item())
        
    plt.plot(losses)
    plt.show()