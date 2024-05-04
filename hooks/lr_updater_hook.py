import torch
from updater import UpdaterBaseHook, annealing_cos, annealing_linear, format_param

class LrUpdaterBaseHook(UpdaterBaseHook):

    def __init__(self, config: dict) -> None:
        print("---------------------------LrUpdaterBaseHook init---------------------------------------")
        super().__init__(config)
        
        self.key = 'lr'

    def _set_value(self, runner, value_groups):
        optim = runner.optimizer if isinstance(runner.optimizer, \
            torch.optim.Optimizer) else runner.optimizer.opt

        for param_group, value in zip(optim.param_groups,
                                      value_groups):
            param_group['lr'] = value

    def _get_warmup_value(self, cur_iters, regular_value):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in regular_value]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                        self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in regular_value]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in regular_value]
        return warmup_lr

    def _before_run(self, runner):
        optim = runner.optimizer if isinstance(runner.optimizer, \
            torch.optim.Optimizer) else runner.optimizer.opt

        for group in optim.param_groups:
            group.setdefault('initial_%s' % self.key, group[self.key])

class OneCycleLrUpdaterHook(LrUpdaterBaseHook):
    def __init__(self, configs: dict) -> None:
        print("----------------------OneCycleLrUpdaterHook init------------------------------------")
        configs['by_epoch'] = False
        super().__init__(configs)

        self._max_lr = configs.get('max_lr')                                                                                            #0.003
        self.total_steps = configs.get('total_steps')                                                                                   #80000
        self.pct_start = configs.get('pct_start')                                                                                       #0.3
        anneal_strategy = configs.get('anneal_strategy')                                                                                #cos
        self.anneal_func = annealing_cos


        self.div_factor = 25 if configs.get('div_factor') is None else configs.get('div_factor')                                        #10
        self.final_div_factor = 1e4 if configs.get('final_div_factor') is None else configs.get('final_div_factor')                     #10000
        self.three_phase = False if configs.get('three_phase') is None else configs.get('three_phase')                                  #Fasle

        self.lr_phases: list = []  # init lr_phases

    def before_run(self, runner):
        optim = runner.optimizer if isinstance(runner.optimizer, \
            torch.optim.Optimizer) else runner.optimizer.opt

        if hasattr(self, 'total_steps'):
            total_steps = self.total_steps
        else:
            total_steps = runner.max_iters

        _max_lr = format_param(type(optim).__name__, optim, self._max_lr)                                                               #[0.003, 0.003]
        
        self.base_value = [lr / self.div_factor for lr in _max_lr]                                                                      #[0.0003, 0.0003]
        for group, lr in zip(optim.param_groups, self.base_value):
            group.setdefault('initial_lr', lr)

        if self.three_phase:
            self.lr_phases.append([float(self.pct_start * total_steps) - 1, 1, self.div_factor])
            self.lr_phases.append([float(2 * self.pct_start * total_steps) - 2, self.div_factor, 1])
            self.lr_phases.append([total_steps - 1, 1, 1 / self.final_div_factor])
        else:
            self.lr_phases.append([float(self.pct_start * total_steps) - 1, 1, self.div_factor])                                        #[23999, 1, 10]
            self.lr_phases.append([total_steps - 1, self.div_factor, 1 / self.final_div_factor])                                        #[79999, 10, 1/10000]

    def get_value(self, runner, base_value: float):
        curr_iter = runner.iter
        start_iter = 0
        for i, (end_iter, start_lr, end_lr) in enumerate(self.lr_phases):
            if curr_iter <= end_iter:
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                lr = self.anneal_func(base_value * start_lr, base_value * end_lr,pct)
                break
            start_iter = end_iter
        return lr


if __name__ == "__main__":
    hook_config = {}
    hook_config['type'] = 'OneCycleLrUpdaterHook'
    hook_config['max_lr'] = 0.003
    hook_config['pct_start'] = 0.3
    hook_config['anneal_strategy'] = 'cos'
    hook_config['div_factor'] = 10
    hook_config['final_div_factor'] = 10000
    hook_config['three_phase'] = False
    hook_config['priority'] = 'VERY_HIGH'
    hook_config['total_steps'] = 80000

    hook = OneCycleLrUpdaterHook(configs=hook_config)

    print(hook.before_run)


