import math
from torch import nn
from torch import Tensor


def swift_(input) -> Tensor:
    '''
    Function type SWIFT Activation
    '''
    out = input.clone()
    under = (out < -1.)
    upper = (1. < out)

    #Linear
    # out[under] = (out[under] + 2.)
    # out[~under * ~upper] = (-out[~under * ~upper])
    # out[upper] = (out[upper] - 2.)

    #SIN
    out[under] = (out[under] + 1.)
    out[~under * ~upper] = (math.pi * (out[~under * ~upper] + 1.)).sin() #-1 <= x <= 1
    out[upper] = (out[upper] - 1.)
    return out

class Swift(nn.Module):
    '''
    Class type SWIFT Activation
    '''
    def __init__(self) -> None:
        super(Swift, self).__init__()

    def forward(self, input) -> Tensor:
        return swift_(input=input)