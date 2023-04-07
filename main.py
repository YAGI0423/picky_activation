import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

import activation

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from logicGateDataset.datasets import XorGate


class Model(Module):
    def __init__(self, shape: list, optimizer, lr: float, loss_function, is_swift: bool=False) -> None:
        super(Model, self).__init__()
        self.layers = nn.Sequential(*self._get_layers(shape, is_swift))

        self.optim = optimizer(self.parameters(), lr=lr)
        self.cri = loss_function
    
    def _get_layers(self, shape: list, is_swift: bool) -> tuple:
        #Mode와 depth에 따른 모델 아키텍처 반환
        act_f = activation.Swift(slope=1.5) if is_swift else nn.ReLU()

        layers = list()
        for d in range(len(shape)-1):
            layers.append(nn.Linear(shape[d], shape[d+1]))
            layers.append(act_f)
        return tuple(layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def train(self, x: Tensor, y: Tensor) -> float:
        y_hat = self.forward(x)
        loss = self.cri(y_hat, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--mode', type=str, default='logic', choices=('logic', 'mnist', 'cifar10'))
    parser.add_argument('--depth', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu', choices=('cuda', 'cpu'))
    # parser.add_argument('--logic', type=str, default='XOR', choices=('AND', 'OR', 'XOR', 'NOT'))
    # parser.add_argument('--input_size', type=int, default=2)
    return parser.parse_args()

def seed_everything(seed: int=423) -> None:
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model_shape(mode: str, depth: int) -> tuple:
    #Mode와 Depth에 따른 모델 형태 반환
    dataset_shapes = {
        #in shape, out shape
        'logic': (2, 4, 6, 1),
        'mnist': (784, 100, 50, 10),
        'cifar10': (64*64, 10),
    }
    shapes = [dataset_shapes[mode][shape] for shape in range(depth)]
    shapes.append(dataset_shapes[mode][-1])
    return tuple(shapes)

def copy_param(alpha: Module, beta: Module) -> None:
    '''
    copy model 'alpha' to 'beta'
    '''
    beta.load_state_dict(alpha.state_dict())

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

def save_plot(swift_acc: list, default_acc: list, head_title: str, figure_path: str) -> None:
    plt.figure(figsize=(14, 5.5))
    plt.suptitle(head_title, fontsize=15, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.125, top=0.825, wspace=0.175, hspace=0.1)
    
    plt.subplot(1, 2, 1)
    plt.title('Swift Activation', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(swift_acc, color='green')
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.title('Default(ReLU) Activation', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(default_acc, color='black')
    plt.grid()
    plt.savefig(figure_path)

get_mean = lambda x: (sum(x) / len(x))


if __name__ == '__main__':
    args = get_args()
    seed_everything() #Fix SEED
    DEVICE = args.device

    model_shape = get_model_shape(mode=args.mode, depth=args.depth)

    swift_model = Model( #swift activation Model
        shape=model_shape,
        optimizer=torch.optim.SGD,
        lr=args.lr,
        loss_function=nn.MSELoss(),
        is_swift=True,
    ).to(DEVICE)
    default_model = Model( #Default(ReLU) activation Model
        shape=model_shape,
        optimizer=torch.optim.SGD,
        lr=args.lr,
        loss_function=nn.MSELoss(),
        is_swift=False,
    ).to(DEVICE)
    
    copy_param(alpha=swift_model, beta=default_model) #swift parameter copy to default model

    if args.mode == 'logic':
        dataLoader = tqdm(
            DataLoader(
                XorGate(dataset_size=1000),
                batch_size=args.batch_size,
                shuffle=True
            )
        )

        swi_losses, def_losses = list(), list()
        for x, y in dataLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            swi_loss = swift_model.train(x, y)
            def_loss = default_model.train(x, y)

            swi_losses.append(swi_loss)
            def_losses.append(def_loss)
            dataLoader.set_description(
                f'Swift Loss: {get_mean(swi_losses):.3f}   Default Loss: {get_mean(def_losses):.3f}'
            )
            
        save_plot(
            swift_acc=swi_losses,
            default_acc=def_losses,
            head_title=f'Loss on XOR Dataset',
            figure_path='./figures/1_lossOnXorDataset.png',
        )

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