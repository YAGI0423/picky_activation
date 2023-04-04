import torch
from torch import nn

from torch import Tensor

from torch.utils.data import DataLoader
from logicGateDataset.datasets import AndGate, OrGate, XorGate, NotGate

from matplotlib import pyplot as plt


class swift(nn.Module):
    def __init__(self, slope: float=1.) -> None:
        super(swift, self).__init__()
        self.slope = slope

    def forward(self, input) -> Tensor:
        out = input.clone()
        upper = (out >= 0.)
    
        out[upper] = (self.slope * out[upper] - 1.)**2 - 1.
        out[~upper] = -(self.slope * out[~upper] + 1.)**2 + 1.
        return out


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.layer1 = nn.Linear(2, 1)
        self.act1 = swift(slope=2.)

    def forward(self, input):
        out = self.act1(self.layer1(input))
        return out


if __name__ == '__main__':
    dataset = XorGate(dataset_size=2000)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Model()

    cri = nn.MSELoss()
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