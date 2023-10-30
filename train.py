import torch
import yaml
import os
import argparse
import pytorch_lightning as pl

from utils import reader, build_labels, build_vocab
from dataset import SequenceDataset
from models import ProtCNN

def train(word2id: dict, 
          fam2label: dict,
          data_dir: str, 
          train_config: dict,
          model_config: dict):
    """Train function """
    train_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "train")
    dev_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "dev")
    test_dataset = SequenceDataset(word2id, fam2label, train_config['seq_max_len'], data_dir, "test")

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
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'], 
        shuffle=False,
        num_workers=train_config['num_workers'],
    )

    num_classes = len(fam2label)
    prot_cnn = ProtCNN(num_classes, train_config, model_config)

    pl.seed_everything(0)
    trainer = pl.Trainer(gpus=train_config['gpu'], max_epochs=train_config['epochs'])
    trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])


if __name__=="__main__":
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(parent_dir, './random_split')

    parser = argparse.ArgumentParser(description='Training hyperparameters')
    parser.add_argument("--config",
                        type=str,
                        default=os.path.join(os.getcwd(), "config.yaml"),
                        help="Configuration yaml")
    
    parser.add_argument("--data_dir",
                        type=str,
                        default=data_dir,
                        help="Random split data directory")
    
    # parser.add_argument("--model_dir",
    #                     type=str,
    #                     default="pretrained_models",
    #                     help="directory containing the models to load")

    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    train_data, train_targets = reader("train", data_dir)
    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    train(word2id, fam2label, args.data_dir, config['training'], config['model'])