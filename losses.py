import torch, torch.nn as nn
import torch.nn.functional as F
import lightning as L

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, x, y):
        return -10 * torch.log10(1 / self.mse(x, y))