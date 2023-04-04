from abc import *

from torch import Tensor
from torch.utils.data import Dataset


class LogicGate(Dataset, metaclass=ABCMeta):
    __constants__ = ('dataset_size, input_size')
    dataset_size: int
    input_size: int
    logic_x: tuple
    logic_y: tuple

    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        self.dataset_size = dataset_size
        self.input_size = input_size
        self.logic_x, self.logic_y = self.__get_logic_table(input_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        table_idx = index % (2 ** self.input_size)
        return Tensor(self.logic_x[table_idx]), Tensor(self.logic_y[table_idx])
    
    def __get_logic_table(self, input_size) -> tuple[tuple, tuple]:
        logic_x, logic_y = list(), list()
        for i in range(2 ** input_size):
            x = f"{format(i, 'b')}".zfill(input_size)
            x = tuple(int(x) for x in x)
            y = self.logicFunction(x)
            
            logic_x.append(x)
            logic_y.append(y)
        return tuple(logic_x), tuple(logic_y)
    
    @abstractmethod
    def logicFunction(self, input: list) -> int:
        '''
        ex) AND, OR, XOR Gate
        '''
        pass


class AndGate(LogicGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(AndGate, self).__init__(dataset_size, input_size)

    def logicFunction(self, input: list) -> tuple:
        return (int(all(input)), )
    

class OrGate(LogicGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(OrGate, self).__init__(dataset_size, input_size)

    def logicFunction(self, input: list) -> tuple:
        return (int(any(input)), )
    

class XorGate(LogicGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(XorGate, self).__init__(dataset_size, input_size)

    def logicFunction(self, input: list) -> tuple:
        flow_logic, *input = input
        for x in input:
            flow_logic ^= x
        return (flow_logic, )
    
class NotGate(LogicGate):
    def __init__(self, dataset_size: int, input_size: int=2) -> None:
        super(NotGate, self).__init__(dataset_size, input_size)

    def logicFunction(self, input: list) -> tuple:
        return tuple(-x + 1 for x in input)
