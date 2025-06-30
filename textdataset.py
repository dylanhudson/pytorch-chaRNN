import random
import torch
import json

class TextDataset:
    
    #seq_length is a hyperparameter that controls the memory context length, and
    #thus also affects efficiency. Longer sequences can result in more context and longer
    #patterns being learned, but requires more memory and compute time. 
    def __init__(self, text_file, seq_length=100):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.seq_length = seq_length

        self.data = [self.char_to_idx[ch] for ch in self.text]

    # Here we create randomly sampled batches of sequences, and put them
    # into tensors. In the "target" tensor, the sequence is shifted by one character
    # from the corresponding input tensor sequence, so we can verify the prediction 
    # for the next character in the sequence.
         
    def get_batch(self, batch_size):
        start_indices = [random.randint(0, len(self.data) - self.seq_length - 1) 
                         for _ in range(batch_size)]
        
        inputs = []
        targets = []

        for start_idx in start_indices:
            input_seq = self.data[start_idx:(start_idx + self.seq_length)]
            target_seq = self.data[(start_idx + 1): (start_idx + self.seq_length + 1)]
            inputs.append(input_seq)
            targets.append(target_seq)

        return torch.tensor(inputs), torch.tensor(targets)
    
    # we need to export the dataset object in a portable format, so we can load it later
    # and use the metadata. 

    def export_dataset_as_json(self, path):
        dataset_info = {
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=4)
        
        print(f'Dataset exported to {path}')


    # alternate constructor to create a skeleton dataset object from the JSON file, so we can load it for inference
    #without requiring the original text file.

    @classmethod
    def load_dataset_from_json(cls, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        dataset = cls.__new__(cls)
        dataset.vocab_size = dataset_info['vocab_size']
        dataset.seq_length = dataset_info['seq_length']
        dataset.char_to_idx = dataset_info['char_to_idx']
        # convert the char_to_idx mapping to int keys, as json loads them as strings
        dataset.idx_to_char = {int(k): v for k, v in dataset_info['idx_to_char'].items()}
        dataset.text = ''  # text is not needed for inference, so we leave it empty
    
        #the idx to char mapping must 

        return dataset
    




