from torch.optim import lr_scheduler
import logging

class OptimSchedulerStep:
    def __init__(self, optimizer, step_size=1000, gamma=0.5):
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimizerScheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)
    
    def step(self, idx=None):
        self.optimizerScheduler.step()
        if idx % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")

class OptimSchedulerGPT:
    def __init__(self, optimizer, batches=2000, verbosity=10):
        self.linear_scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 2.5e-4 * x / batches)
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=batches)
        self.batches = batches
        self.verbosity = verbosity
    
    def step(self, idx):
        if idx < self.batches:
            self.linear_scheduler.step()
        else:
            self.cosine_scheduler.step()
        if idx % (self.batches//self.verbosity) == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")


class CosineSchedule:
    def __init__(self, optimizer, T_max=2000, eta_min=1e-7, verbose=False, verbosity=10):
        self.optimizer = optimizer
        self.T_max = T_max
        self.verbosity = verbosity
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max, eta_min, verbose=verbose
        )

    def step(self, idx):
        self.scheduler.step()
        if idx % (self.T_max//self.verbosity) == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")


class TriangularSchedule:
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, verbosity=10):
        self.scheduler = lr_scheduler.CyclicLR(
            optimizer, base_lr, max_lr, 
            step_size_up=step_size_up, 
            model="triangular"
        )
        self.optimizer = optimizer
        self.step_size_up = step_size_up
        self.verbose = verbosity
    
    def step(self, idx):
        self.scheduler.step()
        if idx % (self.step_size_up//self.verbosity) == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")


class OneCycleSchedule:
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, verbosity=10):
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer, max_lr, 
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            anneal_strategy="linear"
        )
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbosity
    
    def step(self, idx): # TODO: Fix correct reporting
        self.scheduler.step()
        if idx % (self.steps_per_epoch//self.verbosity) == 0:
            for param_group in self.optimizer.param_groups:
                logging.warning(f"[!] Learning rate: {param_group['lr']}")