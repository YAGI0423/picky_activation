import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import Tensor

import activation

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from logicGateDataset.datasets import XorGate


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='logic', choices=('logic', 'mnist', 'cifar10'))
    parser.add_argument('--depth', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--dataset_size', type=int, default=4)
    # parser.add_argument('--logic', type=str, default='XOR', choices=('AND', 'OR', 'XOR', 'NOT'))
    # parser.add_argument('--input_size', type=int, default=2)
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, shape: list, is_swift: bool=False) -> None:
        super(Model, self).__init__()
        self.layers = nn.Sequential(*self.get_layers(shape, is_swift))
    
    def get_layers(self, shape: list, is_swift: bool) -> tuple:
        act_f = activation.Swift(slope=1.) if is_swift else nn.ReLU()

        layers = list()
        for d in range(len(shape)-1):
            layers.append(nn.Linear(shape[d], shape[d+1]))
            layers.append(act_f)
        return tuple(layers)

    def forward(self, input) -> Tensor:
        return self.layers(input)
    
def get_model_shape(mode: str, depth: int) -> tuple:
    dataset_shapes = {
        #in shape, out shape
        'logic': (2, 4, 6, 1),
        'mnist': (784, 100, 50, 10),
        'cifar10': (64*64, 10),
    }
    shapes = [dataset_shapes[mode][shape] for shape in range(depth)]
    shapes.append(dataset_shapes[mode][-1])
    return tuple(shapes)

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
            root='./dataset/mnist/',
            train=train,
            transform=transform,
            download=True
        ), shuffle=train, batch_size=batch_size
    )
    return loader


if __name__ == '__main__':
    args = get_args()

    model_shape = get_model_shape(mode=args.mode, depth=args.depth)
    swift_model = Model(shape=model_shape, is_swift=True)
    defalut_model = Model(shape=model_shape, is_swift=False)

    if args.mode == 'logic':
        dataLoader = DataLoader(XorGate(dataset_size=5000), batch_size=args.batch_size, shuffle=True)

        for x, y in tqdm(dataLoader):
            pass

        raise


    elif args.mode == 'mnist':
        pass
    elif args.mode == 'cifar10':
        pass
    raise
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