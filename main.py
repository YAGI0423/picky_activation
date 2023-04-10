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
from torchvision.datasets import MNIST, CIFAR10
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
        act_f = activation.Swift() if is_swift else nn.ReLU()

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
    parser.add_argument('--mode', type=str, default='cifar10', choices=('logic', 'mnist', 'cifar10'))
    parser.add_argument('--depth', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    return parser.parse_args()

def seed_everything(seed: int=42) -> None:
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_set_info(args: argparse.Namespace) -> None:
    print('\n\n')
    print(f'SETTING INFO'.center(60, '='))
    print(f'+ Mode: {args.mode}({args.device})')
    print(f'+ Depth: {args.depth}', end='\n\n')

def get_argsByMode(mode: str) -> tuple:
    '''
    mode에 따른 하이퍼파라미터 반환
    (1) LOGIC
        batch size: 1
        lr: 0.01
        loss fc: MSE

    (2) MNIST
        batch size: 32
        lr: 0.0001
        loss fc: cross entropy

    (3) CIFAR 10
        batch size: 128
        lr: 0.001
        loss fc: cross entropy
    '''
    if mode == 'logic':
        return 1, 0.01, nn.MSELoss() #batch, lr, lossFC
    elif mode == 'mnist':
        return 4, 0.0001, nn.CrossEntropyLoss()
    elif mode == 'cifar10':
        return 128, 0.0001, nn.CrossEntropyLoss()
    raise

def get_model_shape(mode: str, depth: int) -> tuple:
    #Mode와 Depth에 따른 모델 형태 반환
    dataset_shapes = {
        #in shape, out shape
        'logic': (2, 4, 6, 1),
        'mnist': (784, 100, 50, 10),
        'cifar10': (3*32*32, 1000, 100, 10),
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

def cifarDataLoader(train: bool, batch_size: int) -> DataLoader:
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        Lambda(lambda x: torch.flatten(x)),
    ])
    loader = DataLoader(
        CIFAR10(
            root='./dataset/cifar10/',
            train=train,
            transform=transform,

        ), shuffle=train, batch_size=batch_size
    )
    return loader

def save_plot(swift_loss: list, default_loss: list, head_title: str, figure_path: str, ylim: tuple=(0, 1)) -> None:
    plt.figure(figsize=(14, 5.5))
    plt.suptitle(head_title, fontsize=15, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.125, top=0.825, wspace=0.175, hspace=0.1)
    
    plt.subplot(1, 2, 1)
    plt.title('Swift Activation', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(swift_loss, color='green')
    plt.grid()
    plt.ylim(ylim)
    
    plt.subplot(1, 2, 2)
    plt.title('Default(ReLU) Activation', fontdict={'fontsize': 13, 'fontweight': 'bold'}, loc='left', pad=10)
    plt.plot(default_loss, color='black')
    plt.grid()
    plt.ylim(ylim)
    plt.savefig(figure_path)

get_mean = lambda x: (sum(x) / len(x))


if __name__ == '__main__':
    args = get_args()
    seed_everything() #Fix SEED
    DEVICE = args.device
    BATCH_SIZE, LR, CRI = get_argsByMode(mode=args.mode)
    model_shape = get_model_shape(mode=args.mode, depth=args.depth)
    print_set_info(args)

    swift_model = Model( #swift activation Model
        shape=model_shape,
        optimizer=torch.optim.Adam,
        lr=LR,
        loss_function=CRI,
        is_swift=True,
    ).to(DEVICE)
    default_model = Model( #Default(ReLU) activation Model
        shape=model_shape,
        optimizer=torch.optim.Adam,
        lr=LR,
        loss_function=CRI,
        is_swift=False,
    ).to(DEVICE)
    
    copy_param(alpha=swift_model, beta=default_model) #swift parameter copy to default model

    if args.mode == 'logic':
        dataLoader = tqdm(
            DataLoader(
                XorGate(dataset_size=1000),
                batch_size=BATCH_SIZE,
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
            swift_loss=swi_losses,
            default_loss=def_losses,
            head_title=f'Loss on XOR Dataset',
            figure_path='./figures/1_lossOnXorDataset.png',
        )
    elif args.mode == 'mnist':
        swi_te_losses, def_te_losses = list(), list() #TEST
        for e in range(3): #EPOCH
            train_dataLoader = tqdm(mnistDataLoader(train=True, batch_size=BATCH_SIZE))  
            swi_tr_losses, def_tr_losses = list(), list() #TRAIN  
            for x, y in train_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                swi_loss = swift_model.train(x, y)
                def_loss = default_model.train(x, y)

                swi_tr_losses.append(swi_loss)
                def_tr_losses.append(def_loss)
                train_dataLoader.set_description(
                    f'TRAIN   Swift Loss: {get_mean(swi_tr_losses):.3f}   Default Loss: {get_mean(def_tr_losses):.3f}'
                )

            test_dataLoader = tqdm(mnistDataLoader(train=False, batch_size=10000)) #One Batch All Data
            for x, y in test_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                swi_loss = swift_model.cri(swift_model(x), y).item()
                def_loss = default_model.cri(default_model(x), y).item()

                swi_te_losses.append(swi_loss)
                def_te_losses.append(def_loss)
                test_dataLoader.set_description(
                    f'TEST   Swift Loss: {swi_loss:.3f}   Default Loss: {def_loss:.3f}'
                )
            print()

        save_plot(
            swift_loss=swi_te_losses,
            default_loss=def_te_losses,
            head_title=f'Loss on MNIST Test set',
            figure_path='./figures/2_lossOnMNIST_Test_set.png',
            ylim=(0.2, 0.6),
        )

    elif args.mode == 'cifar10':
        swi_te_losses, def_te_losses = list(), list() #TEST
        for e in range(10): #EPOCH
            train_dataLoader = tqdm(cifarDataLoader(train=True, batch_size=BATCH_SIZE))  
            swi_tr_losses, def_tr_losses = list(), list() #TRAIN  
            for x, y in train_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                swi_loss = swift_model.train(x, y)
                def_loss = default_model.train(x, y)

                swi_tr_losses.append(swi_loss)
                def_tr_losses.append(def_loss)
                train_dataLoader.set_description(
                    f'TRAIN   Swift Loss: {get_mean(swi_tr_losses):.3f}   Default Loss: {get_mean(def_tr_losses):.3f}'
                )

            test_dataLoader = tqdm(cifarDataLoader(train=False, batch_size=12288)) #One Batch All Data
            for x, y in test_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                swi_loss = swift_model.cri(swift_model(x), y).item()
                def_loss = default_model.cri(default_model(x), y).item()

                swi_te_losses.append(swi_loss)
                def_te_losses.append(def_loss)
                test_dataLoader.set_description(
                    f'TEST   Swift Loss: {swi_loss:.3f}   Default Loss: {def_loss:.3f}'
                )
            print()

        save_plot(
            swift_loss=swi_te_losses,
            default_loss=def_te_losses,
            head_title=f'Loss on MNIST Test set',
            figure_path='./figures/3_lossOnCIFAR10_Test_set.png',
            ylim=(1., 2.),
        )
