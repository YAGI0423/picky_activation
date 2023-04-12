from torch import nn
from torch import Tensor


def picky_(input) -> Tensor:
    '''
    Function type SWIFT Activation
    '''
    out = input.clone()

    #under picky <BEST
    under = (out < 0.)
    out[under] = (-out[under])
    out[~under] = (out[~under])
    return out

class Picky(nn.Module):
    '''
    Class type SWIFT Activation
    '''
    def __init__(self) -> None:
        super(Picky, self).__init__()

    def forward(self, input) -> Tensor:
        return picky_(input=input)