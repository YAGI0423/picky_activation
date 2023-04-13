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
    def __init__(self, shape: list, optimizer, lr: float, loss_function, is_picky: bool=False) -> None:
        super(Model, self).__init__()
        self.layers = nn.Sequential(*self._get_layers(shape, is_picky))

        self.optim = optimizer(self.parameters(), lr=lr)
        self.cri = loss_function
    
    def _get_layers(self, shape: list, is_picky: bool) -> tuple:
        #Mode와 depth에 따른 모델 아키텍처 반환
        act_f = activation.Picky() if is_picky else nn.ReLU()

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
    parser.add_argument('--mode', type=str, default='logic', choices=('logic', 'mnist', 'cifar10', 'AE'))
    parser.add_argument('--depth', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--device', type=str, default='cpu', choices=('cuda', 'cpu'))
    return parser.parse_args()

def seed_everything(seed: int=423) -> None:
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
    print(f'+ Depth: {args.depth}')
    print('=' * 60, end='\n\n')

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

    (4) AutoEncoder
        batch size: 16
        lr: 0.001
        loss fc: MSE
    '''
    if mode == 'logic':
        return 1, 0.01, nn.MSELoss() #batch, lr, lossFC
    elif mode == 'mnist':
        return 4, 0.0001, nn.CrossEntropyLoss()
    elif mode == 'cifar10':
        return 128, 0.0001, nn.CrossEntropyLoss()
    elif mode == 'AE':
        return 256, 0.0005, nn.MSELoss()
    raise

def get_model_shape(mode: str, depth: int) -> tuple:
    #Mode와 Depth에 따른 모델 형태 반환
    dataset_shapes = {
        #in shape, out shape
        'logic': (2, 4, 6, 1),
        'mnist': (784, 100, 50, 10),
        'cifar10': (3*32*32, 1000, 100, 10),
        'AE': (3*32*32, 1024, 1024, 3*32*32),
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
            root='./datasets/mnist/',
            train=train,
            transform=transform,
            download=True
        ), shuffle=train, batch_size=batch_size
    )
    return loader

def cifarDataLoader(train: bool, batch_size: int) -> DataLoader:
    is_donwload = not os.path.isdir('./datasets/cifar10') #isdir -> False, nodir -> True

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        Lambda(lambda x: torch.flatten(x)),
    ])
    loader = DataLoader(
        CIFAR10(
            root='./datasets/cifar10/',
            train=train,
            transform=transform,
            download=is_donwload,
        ), shuffle=train, batch_size=batch_size
    )
    return loader

def save_plot(picky_loss: list, default_loss: list, head_title: str, figure_path: str, ylim: tuple=(0, 1)) -> None:
    plt.figure(figsize=(7, 4))
    plt.suptitle(head_title, fontsize=15, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.125, top=0.85, wspace=0.175, hspace=0.1)
    
    plt.plot(default_loss, color='gray', label='Default(ReLU) Activation')
    plt.plot(picky_loss, color='green', label='Picky Activtaion')
    plt.legend()
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

    picky_model = Model( #picky activation Model
        shape=model_shape,
        optimizer=torch.optim.Adam,
        lr=LR,
        loss_function=CRI,
        is_picky=True,
    ).to(DEVICE)
    default_model = Model( #Default(ReLU) activation Model
        shape=model_shape,
        optimizer=torch.optim.Adam,
        lr=LR,
        loss_function=CRI,
        is_picky=False,
    ).to(DEVICE)
    
    copy_param(alpha=picky_model, beta=default_model) #picky parameter copy to default model

    if args.mode == 'logic':
        dataLoader = tqdm(
            DataLoader(
                XorGate(dataset_size=300),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        )

        pick_losses, def_losses = list(), list()
        for x, y in dataLoader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pick_loss = picky_model.train(x, y)
            def_loss = default_model.train(x, y)

            pick_losses.append(pick_loss)
            def_losses.append(def_loss)
            dataLoader.set_description(
                f'Picky Loss: {get_mean(pick_losses):.3f}   Default Loss: {get_mean(def_losses):.3f}'
            )
            
        save_plot(
            picky_loss=pick_losses,
            default_loss=def_losses,
            head_title=f'Loss on XOR Dataset',
            figure_path='./figures/1_lossOnXorDataset.png',
        )
    elif args.mode == 'mnist':
        pick_te_losses, def_te_losses = list(), list() #TEST
        for e in range(3): #EPOCH
            train_dataLoader = tqdm(mnistDataLoader(train=True, batch_size=BATCH_SIZE))  
            pick_tr_losses, def_tr_losses = list(), list() #TRAIN  
            for x, y in train_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pick_loss = picky_model.train(x, y)
                def_loss = default_model.train(x, y)

                pick_tr_losses.append(pick_loss)
                def_tr_losses.append(def_loss)
                train_dataLoader.set_description(
                    f'TRAIN   Picky Loss: {get_mean(pick_tr_losses):.3f}   Default Loss: {get_mean(def_tr_losses):.3f}'
                )

            test_dataLoader = tqdm(mnistDataLoader(train=False, batch_size=10000)) #One Batch All Data
            for x, y in test_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pick_loss = picky_model.cri(picky_model(x), y).item()
                def_loss = default_model.cri(default_model(x), y).item()

                pick_te_losses.append(pick_loss)
                def_te_losses.append(def_loss)
                test_dataLoader.set_description(
                    f'TEST   Picky Loss: {pick_loss:.3f}   Default Loss: {def_loss:.3f}'
                )
            print()

        save_plot(
            picky_loss=pick_te_losses,
            default_loss=def_te_losses,
            head_title=f'Loss on MNIST Test set',
            figure_path='./figures/2_lossOnMNIST_Test_set.png',
            ylim=(0.2, 0.6),
        )
    elif args.mode == 'cifar10':
        pick_te_losses, def_te_losses = list(), list() #TEST
        for e in range(15): #EPOCH
            train_dataLoader = tqdm(cifarDataLoader(train=True, batch_size=BATCH_SIZE))  
            pick_tr_losses, def_tr_losses = list(), list() #TRAIN  
            for x, y in train_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pick_loss = picky_model.train(x, y)
                def_loss = default_model.train(x, y)

                pick_tr_losses.append(pick_loss)
                def_tr_losses.append(def_loss)
                train_dataLoader.set_description(
                    f'TRAIN   Picky Loss: {get_mean(pick_tr_losses):.3f}   Default Loss: {get_mean(def_tr_losses):.3f}'
                )

            test_dataLoader = tqdm(cifarDataLoader(train=False, batch_size=12288)) #One Batch All Data
            for x, y in test_dataLoader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pick_loss = picky_model.cri(picky_model(x), y).item()
                def_loss = default_model.cri(default_model(x), y).item()

                pick_te_losses.append(pick_loss)
                def_te_losses.append(def_loss)
                test_dataLoader.set_description(
                    f'TEST   Picky Loss: {pick_loss:.3f}   Default Loss: {def_loss:.3f}'
                )
            print()

        save_plot(
            picky_loss=pick_te_losses,
            default_loss=def_te_losses,
            head_title=f'Loss on CIFAR10 Test set',
            figure_path='./figures/3_lossOnCIFAR10_Test_set.png',
            ylim=(1., 2.),
        )
    elif args.mode == 'AE':
        clip_img = lambda img: (img.min() - img) / (img.min() - img.max())
        reshape_img = lambda img: img.view(3, 32, 32).permute(1, 2, 0)
        recover_img = lambda img: reshape_img(clip_img(img))


        pick_te_losses, def_te_losses = list(), list() #TEST
        for e in range(15): #EPOCH
            train_dataLoader = tqdm(cifarDataLoader(train=True, batch_size=BATCH_SIZE))
            pick_tr_losses, def_tr_losses = list(), list() #TRAIN  
            for x, _ in train_dataLoader:
                x = x.to(DEVICE)

                pick_loss = picky_model.train(x, x)
                def_loss = default_model.train(x, x)

                pick_tr_losses.append(pick_loss)
                def_tr_losses.append(def_loss)
                train_dataLoader.set_description(
                    f'TRAIN   Picky Loss: {get_mean(pick_tr_losses):.3f}   Default Loss: {get_mean(def_tr_losses):.3f}'
                )

            test_dataLoader = tqdm(cifarDataLoader(train=False, batch_size=12288))
            for x, _ in test_dataLoader:
                x = x.to(DEVICE)
                pick_loss = picky_model.cri(picky_model(x), x).item()
                def_loss = default_model.cri(default_model(x), x).item()

                pick_te_losses.append(pick_loss)
                def_te_losses.append(def_loss)
                test_dataLoader.set_description(
                    f'TEST   Picky Loss: {pick_loss:.3f}   Default Loss: {def_loss:.3f}'
                )
            print()

        save_plot(
            picky_loss=pick_te_losses,
            default_loss=def_te_losses,
            head_title=f'Loss on CIFAR10 Test set for Auto Encoder',
            figure_path='./figures/4_lossOnCIFAR10_Test_set_AE.png',
            ylim=None,
        )

        samples, _ = next(iter(cifarDataLoader(train=False, batch_size=5)))
        
        plt.figure(figsize=(10, 6))
        plt.suptitle('Auto Encoder Output on CIFAR-10', fontsize=15, fontweight='bold')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.88, wspace=0.175, hspace=0.1)
        

        pick_samples = picky_model(samples.to(DEVICE)).detach().cpu()
        def_samples = default_model(samples.to(DEVICE)).detach().cpu()
        for row, (sp, pick_sp, def_sp) in enumerate(zip(samples, pick_samples, def_samples), 1):

            plt.subplot(3, 5, row)
            plt.axis('off')
            plt.imshow(recover_img(sp))
            if row == 1: plt.text(-16, 16, 'Input', fontdict={'weight': 'bold', 'size': 14})

            plt.subplot(3, 5, 5+row)
            plt.axis('off')
            plt.imshow(recover_img(pick_sp))
            if row == 1: plt.text(-16, 16, 'Picky' ,fontdict={'weight': 'bold', 'size': 14, 'color': 'blue'})

            plt.subplot(3, 5, 10+row)
            plt.axis('off')
            plt.imshow(recover_img(def_sp))
            if row == 1: plt.text(-16, 16, 'ReLU', fontdict={'weight': 'bold', 'size': 14})
        plt.savefig('./figures/5_AE_OutputOnCifar10')
