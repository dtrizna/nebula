from torch.optim import lr_scheduler
import logging


class OptimSchedulerStep:
    def __init__(self, optimizer, step_size=500, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        self.optimizerScheduler = lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
    
    def step(self, idx=None):
        self.optimizerScheduler.step()
        if idx % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")

class OptimSchedulerGPT:
    def __init__(self, optimizer, nr=2000):
        self.linear_scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 2.5e-4 * x / nr)
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr)
        self.nr = nr
    
    def step(self, idx):
        if idx < self.nr:
            self.linear_scheduler.step()
        else:
            self.cosine_scheduler.step()
        if idx % self.step_size == 100:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")
