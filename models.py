import pytorch_lightning as pl
import torch
import math
import torch.nn.functional as F
import torchmetrics

from losses import FocalLoss

class PositionalEncoding(torch.nn.Module):
    """
    Positional Embedding using sin/cos of different frequencies.
    
    Args:
        seq_len (int): Input sequence length
        dropout: Probablility of dropout
        max_len: Hyperparm to geenrate embedding 
    """

    def __init__(self, seq_len: int=120, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, seq_len, 2) * (-math.log(10000.0) / seq_len))
        pe = torch.zeros(max_len, 1, seq_len)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class ProtTransformer(torch.nn.Module):
    """Protein Transformer Encoder Module (similar to BERT)."""

    def __init__(self, num_classes: int, model_config: dict, vocab_size: int=22):
        super().__init__()

        self.seq_len = model_config['seq_max_len']
        # self.pos_encoder = PositionalEncoding( model_config['seq_max_len'])
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model= model_config['seq_max_len'], nhead= model_config['n_head'], batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=model_config['n_layers'])
        self.linear = torch.nn.Linear(model_config['linear_input_channels'], num_classes)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x) -> torch.tensor:
        # x = x.permute((1, 0, 2))
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x =  x.flatten(start_dim=1)
        return self.linear(x)
    

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

class ProtCNN(torch.nn.Module):
    """ProtCNN model.
    
    Args:
        num_classes (int): Num of labels in training data
        train_config (dict): Training Configuration
        model_config (dict): Model Configuration
        vocab_size (int): Vocabulary size of training data
    """

    def __init__(self, num_classes: int, model_config: dict, vocab_size: int=22):
        super().__init__() 

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(vocab_size, model_config['resblock_channels'], kernel_size=1, padding=0, bias=False),
            ResidualBlock(model_config['resblock_channels'], model_config['resblock_channels'], dilation=2),
            ResidualBlock(model_config['resblock_channels'], model_config['resblock_channels'], dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(model_config['linear_input_channels'], num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

class ProteinClassifier(pl.LightningModule):
    """Protein Classification wrapper for Pytorch Lightning Module class.
    
    Args:
        num_classes (int): Num of labels in training data
        train_config (dict): Training Configuration
        model_config (dict): Model Configuration
        vocab_size (int): Vocabulary size of training data
    """

    def __init__(self, num_classes: int, train_config: dict, model_config: dict) -> None:
        super().__init__()

        if model_config['type'] == 'Transformer':
            self.model = ProtTransformer(num_classes, model_config)
        elif model_config['type'] == "CNN":
            self.model = ProtCNN(num_classes, model_config)

        self.save_hyperparameters("num_classes", "train_config", "model_config")
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.train_config = train_config
        self.model_config = model_config
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x.float())
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.tensor:
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        focal_loss = FocalLoss()
        loss = focal_loss(y_hat, y)
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