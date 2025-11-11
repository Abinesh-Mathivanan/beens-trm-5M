import copy
import torch.nn as nn


class EMAHelper:
    def __init__(self, mu=0.999): 
        self.mu, self.shadow = mu, {}

    def register(self, module):
        module = module.module if isinstance(module, nn.DataParallel) else module
        for name, param in module.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()

    def update(self, module):
        module = module.module if isinstance(module, nn.DataParallel) else module
        for name, param in module.named_parameters():
            if param.requires_grad: self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        module_copy = module_copy.module if isinstance(module_copy, nn.DataParallel) else module_copy
        for name, param in module_copy.named_parameters():
            if param.requires_grad: param.data.copy_(self.shadow[name].data)
        return module_copy