import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F


class Lambda(torch.nn.Module):
    def __init__(self, func):  
        """Lambda wrapper class for function."""
        super().__init__()
        self.func = func

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int=1) -> None:
        super().__init__()   
        
        # Initialize the required layers
        self.skip = torch.nn.Sequential()
            
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, bias=False, padding=1)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))
        
        return x2 + self.skip(x)
    

class ProtCNN(pl.LightningModule):
    """ProtCNN model class.
    
    Args:
        num_classes:
        resblock_size:
    """
    def __init__(self, num_classes: int, train_config, model_config) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, model_config['resblock_channels'], kernel_size=1, padding=0, bias=False),
            ResidualBlock(model_config['resblock_channels'], model_config['resblock_channels'], dilation=2),
            ResidualBlock(model_config['resblock_channels'], model_config['resblock_channels'], dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(model_config['linear_input_channels'], num_classes)
        )
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.train_config = train_config
        self.model_config = model_config
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x.float())
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.tensor:
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torchmetrics.Accuracy:
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)        
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)
        return acc
        
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.train_config['optimizer']['lr'], 
                                    momentum=self.train_config['optimizer']['momentum'], 
                                    weight_decay=self.train_config['optimizer']['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.train_config['lr_scheduler']['milestones'], 
                                                            gamma=self.train_config['lr_scheduler']['gamma'])

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }