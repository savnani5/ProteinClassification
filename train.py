import argparse
import os

import pytorch_lightning as pl
import torch
import torchmetrics
import yaml

from dataset import SequenceDataset
from models import ProtCNN
from utils import build_labels, build_vocab, reader


def train(word2id: dict, 
          fam2label: dict,
          data_dir: str, 
          train_config: dict,
          model_config: dict,
          eval: bool=False,
          chkpt_dir: str=None):
    """Train function."""
    if not eval:
        # Training mode
        print("Training the model............")
        train_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "train")
        dev_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "dev")

        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_config['batch_size'], 
            shuffle=True,
            num_workers=train_config['num_workers'],
        )
        dataloaders['dev'] = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=train_config['batch_size'], 
            shuffle=False,
            num_workers=train_config['num_workers'],
        )

        num_classes = len(fam2label)
        prot_cnn = ProtCNN(num_classes, train_config, model_config)
        pl.seed_everything(0)
        accelerator = None 
        if torch.cuda.is_available():
            accelerator = 'gpu'
            trainer = pl.Trainer(accelerator=accelerator, gpus=train_config['gpu'] ,max_epochs=train_config['epochs'])
        else:
            trainer = pl.Trainer(accelerator=accelerator, max_epochs=train_config['epochs'])
        trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])
    
    else:
        # Evaluation mode
        print("Evaluating the model on test dataset............")
        test_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "test")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_config['batch_size'],  # Can change for evaluation
            shuffle=False,
            num_workers=train_config['num_workers'],
        )
        model = ProtCNN.load_from_checkpoint(chkpt_dir)
        model.eval()
        test_acc = torchmetrics.Accuracy()
        for batch in test_dataloader:
            x, y = batch['sequence'], batch['target']
            y_hat = model(x)
            pred = torch.argmax(y_hat, dim=1)        
            # print(pred, y)
            acc = test_acc(pred, y)
            print('test_acc per batch', acc)


if __name__=="__main__":
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(parent_dir, './random_split')

    parser = argparse.ArgumentParser(description='Training hyperparameters')
    parser.add_argument("--config",
                        type=str,
                        default=os.path.join(os.getcwd(), "config.yaml"),
                        help="Configuration yaml path")
    
    parser.add_argument("--data_dir",
                        type=str,
                        default=data_dir,
                        help="Random split data path")
    
    parser.add_argument("-e",
                        action='store_true',
                        help="Evaluation flag, pass to run in eval mode")

    parser.add_argument("-c",
                        type=str,
                        default="",
                        help="Pass checkpoint path if using eval mode")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    train_data, train_targets = reader("train", data_dir)
    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    if args.e:
        eval=True
        chkpt_dir=args.c
    else:
        eval=False
        chkpt_dir=""

    train(word2id, 
          fam2label, 
          args.data_dir, 
          config['training'], 
          config['model'], 
          eval, 
          chkpt_dir)