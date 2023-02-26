from torch.optim import lr_scheduler
import logging

VERBOSITY=1000

class BaseScheduler:
    def __init__(self, optimizer, verbosity=VERBOSITY):
        self.optimizer = optimizer
        self.verbosity = verbosity
        self.scheduler = None
    
    def step(self, idx=None):
        assert self.scheduler is not None, "Scheduler not initialized"
        self.scheduler.step()
        if self.verbosity and idx % self.verbosity == 0:
            logging.warning(f"[!] Learning rate: {self.optimizer.param_groups[0]['lr']}")
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class OptimSchedulerStep(BaseScheduler):
    def __init__(self, optimizer, step_size, gamma=0.5, verbosity=VERBOSITY):
        super().__init__(optimizer, verbosity)
        self.step_size = step_size
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)
    
class OptimSchedulerGPT(BaseScheduler):
    def __init__(self, optimizer, max_lr, half_cycle_batches, verbosity=VERBOSITY):
        super().__init__(optimizer, verbosity)
        self.linear_scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: max_lr * x / half_cycle_batches)
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=half_cycle_batches, eta_min=max_lr / 10)
        self.half_cycle_batches = half_cycle_batches
    
    def step(self, idx):
        if idx < self.half_cycle_batches:
            self.linear_scheduler.step()
        else:
            self.cosine_scheduler.step()
        if idx % self.verbosity == 0:
            logging.warning(f"[!] Learning rate: {self.optimizer.param_groups[0]['lr']}")

class CosineSchedule(BaseScheduler):
    def __init__(self, optimizer, T_max, eta_min, verbose=False, verbosity=VERBOSITY):
        super().__init__(optimizer, verbosity)
        self.T_max = T_max
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max, eta_min, verbose=verbose
        )

class TriangularSchedule(BaseScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, verbosity=VERBOSITY):
        super().__init__(optimizer, verbosity)
        self.scheduler = lr_scheduler.CyclicLR(
            optimizer, base_lr, max_lr, 
            step_size_up=step_size_up, 
            mode="triangular",
            cycle_momentum=False
        )
        self.step_size_up = step_size_up

class OneCycleSchedule(BaseScheduler):
    def __init__(self, optimizer, max_lr, total_steps, verbosity=VERBOSITY):
        super().__init__(optimizer, verbosity)
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr, 
            total_steps,
            anneal_strategy="linear"
        )
        self.total_steps = total_steps
