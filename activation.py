from torch import nn
from torch import Tensor


def swift_(input, slope: float=1.) -> Tensor:
    '''
    Function type SWIFT Activation
    '''
    out = input.clone()
    upper = (out >= 0.)

    out[upper] = (slope * out[upper] - 1.)**2 - 1.
    out[~upper] = -(slope * out[~upper] + 1.)**2 + 1.
    return out

class Swift(nn.Module):
    '''
    Class type SWIFT Activation
    '''
    def __init__(self, slope: float=1.) -> None:
        super(Swift, self).__init__()
        self.slope = slope

    def forward(self, input) -> Tensor:
        return swift_(input=input, slope=self.slope)