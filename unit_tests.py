import os
import yaml
import pytest
import torch

from dataset import SequenceDataset
from models import ProteinClassifier
from utils import build_labels, build_vocab, reader


class TestClass:
    """Unit tests class to test different modules in code. Can be extended to 
    multiple files if there are sufficient tests per module.
    """
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(parent_dir, './random_split')
    test_data, test_targets = reader("dev", data_dir)
    word2id = build_vocab(test_data)
    fam2label = build_labels(test_targets)
    seq_max_len = 120
    batch_size = 4
    split = "dev"

    @pytest.fixture(scope="class")
    def dataset(self):
        return SequenceDataset(self.word2id, self.fam2label, self.seq_max_len, self.data_dir, self.split)
    
    def test_Sequence_data(self, dataset):
        output_shape = next(iter(dataset))['sequence'].shape
        assert output_shape == torch.zeros((22, 120)).shape, (f"Ouput shape {output_shape} does not match expected shape.")

    def test_Sequence_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(dataloader))
        assert batch['sequence'].shape == torch.zeros((4, 22, 120)).shape, (f"Ouput shape {batch['sequence'].shape} does not match expected shape.")
        assert batch['target'].shape == torch.zeros((4)).shape, (f"Ouput shape {batch['target'].shape} does not match expected shape.")

    # @pytest
    def test_model(self, dataset):
        config_file = os.path.join(os.getcwd(), "config.yaml")
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        num_classes = len(self.fam2label)
        prot_cnn = ProteinClassifier(num_classes, config['training'], config['model'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
        )
        input = next(iter(dataloader))['sequence']
        output_shape = prot_cnn(input).shape
        assert output_shape == torch.zeros((self.batch_size, num_classes)).shape, (f"Ouput shape {output_shape} does not match expected shape.")