import numpy as np


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001, warmup=3):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup  # 前N个epoch不触发早停
        self.counter = 0
        self.best_loss = np.inf
        self.epoch = 0

    def should_stop(self, current_loss):
        self.epoch += 1
        if self.epoch <= self.warmup:  # 热身期不触发
            return False

        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False