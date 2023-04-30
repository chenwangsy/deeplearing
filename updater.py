import numbers
import torch

from abc import ABC, abstractmethod
from math import cos, pi
from typing import Union, overload
from hook import Hook

class Updater(ABC):
    support_warmup = ['constant', 'linear', 'exp']

    def __init__(self, config: dict):
        print("----------------------------Updater init----------------------------")
        self._configure(config)

    def _configure(self, config: dict):
        warmup = config.get('warmup')                                                                                   #None
        warmup_iters = 0 if config.get('warmup_iters') is None else config.get('warmup_iters')                          #0
        warmup_ratio = 0.1 if config.get('warmup_ratio') is None else config.get('warmup_ratio')                        #0.1


        if warmup is not None:
            if warmup not in Updater.support_warmup:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        self.warmup = warmup                                                                                            #None
        self.warmup_iters = warmup_iters                                                                                #0
        self.warmup_ratio = warmup_ratio                                                                                #0.1

        self.by_epoch = True if config.get('by_epoch') is None else config['by_epoch']                                  #False
        self.warmup_by_epoch = False if config.get('warmup_by_epoch') is None else config['warmup_by_epoch']            #False

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_value: Union[list, dict] = []  # initial value for all param groups
        self.regular_value: list = []  # expected value if no warming up is performed

    @abstractmethod
    @overload
    def get_value(self, runner, base_value: float):
        pass
    
    @abstractmethod
    def _set_value(self, runner, value_groups):
        pass

    @abstractmethod
    def _get_warmup_value(self, cur_iters: int, regular_value: float):
        pass

    def get_regular_value(self, runner):
        return [self.get_value(runner, _base_value) for _base_value in self.base_value]

    def get_warmup_value(self, cur_iters: int):
        return self._get_warmup_value(cur_iters, self.regular_value)

class UpdaterBaseHook(Hook, Updater):
    def __init__(self, config: dict) -> None:
        print("--------------------------------UpdaterBaseHook init-------------------------------")
        assert config.get('priority') is not None
        super().__init__(config['priority'])
        Updater.__init__(self, config)

        self.key = None

    @abstractmethod
    def _before_run(self, runner):
        pass

    def before_run(self, runner):
        self._before_run(runner)

        optim = runner.optimizer if isinstance(runner.optimizer, \
            torch.optim.Optimizer) else runner.optimizer.opt
        self.base_value = [
            group['initial_%s' % self.key] for group in optim.param_groups
        ]

    def before_train_epoch(self, runner):
        """OneCycleLrUpdaterHook什么也不干"""
        if self.warmup_iters is None:                                               #判断过不了，直接跳过
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len
        
        if not self.by_epoch:                                                       #OneCycleLrUpdaterHook运行这个，因此
            return
        self.regular_value = self.get_regular_value(runner)
        self._set_value(runner, self.regular_value)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:                                                       #OneCycleLrUpdaterHook运行这个
            self.regular_value = self.get_regular_value(runner)                     #[lr1, lr2]
            if self.warmup is None or cur_iter >= self.warmup_iters:                #OneCycleLrUpdaterHook运行这个
                self._set_value(runner, self.regular_value)
            else:
                warmup_value = self.get_warmup_value(cur_iter)
                self._set_value(runner, warmup_value)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_value(runner, self.regular_value)
            else:
                warmup_value = self.get_warmup_value(cur_iter)
                self._set_value(runner, warmup_value)

def annealing_cos(start: float, end: float, factor: float, weight: float = 1) -> float:
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out

def annealing_linear(start: float, end: float, factor: float) -> float:
    return start + (end - start) * factor

def format_param(name, optim, param):
    if isinstance(param, numbers.Number):
        return [param] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f'expected {len(optim.param_groups)} '
                             f'values for {name}, got {len(param)}')
        return param
    else:  # multi optimizers
        if name not in param:
            raise KeyError(f'{name} is not found in {param.keys()}')
        return param[name]
