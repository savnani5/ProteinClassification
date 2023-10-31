import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    """Custom Focal loss class for imblanced data distribution.
    Args:
        weight (float): alpha constant for focal loss
        gamma (float): gamma constant for focal loss
        reduction (str): Operation to perform on weighted loss"""
    
    def __init__(self, weight: float=None, gamma: float=2, reduction: str='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss