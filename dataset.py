import numpy as np
import torch

from utils import reader


class SequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset for protein sequences.
    
    Args:
        word2id (dict): Dictionary to map letters to ids
        fam2label (dict): Dictionary to map family to class labels
        max_len (int) : maximum lenght of input
        data_path (str): data path to read data
        split (str): train/test/dev split
    """

    def __init__(self, word2id: dict, fam2label: dict, max_len: int, data_path: str, split: str) -> None:
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        
        self.data, self.label = reader(split, data_path)
        
    def __len__(self):
        """Length of dataset."""
        return len(self.data)

    def __getitem__(self, index: int)-> dict:
        """Return dictonary of sequene and label from given index
        Args:
            index (int): Sample index in dataset
        
        Returns:
            dict: Dictonary of sequence and label
        """
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
       
        return {'sequence': seq, 'target' : label}
    
    def preprocess(self, text: str) -> torch.tensor:
        """Preprocess the text and convert input to one hot encodings.
        Args:
            text (str): Input text from dataframe index
        
        Returns:
            tensor: One hot tensor of the text sequence   
        """
        seq = []
        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))
                
        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]
                
        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq, dtype=np.int64))
        
        # One-hot encode    
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id)) 

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1,0)

        return one_hot_seq