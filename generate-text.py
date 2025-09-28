import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse

from textdataset import TextDataset
from textrnn import TextRNN
    

def generate_text(model: TextRNN, dataset: TextDataset, device: str, seed_text='', length=100, temperature=1.0) -> str:
    model.eval()

    if not seed_text:
        seed_text = random.choice(dataset.text)
    
    current_seq = [dataset.char_to_idx.get(ch, 0) for ch in seed_text]
    generated = seed_text

    with torch.no_grad():
        hidden = None

        for _ in range(length):

            if len(current_seq) > dataset.seq_length:
                current_seq = current_seq[-dataset.seq_length:]
                
            input_tensor = torch.tensor([current_seq]).to(device)
            
            output, hidden = model(input_tensor, hidden)

            last_output = output[-1] / temperature
            probabilities = torch.softmax(last_output, dim=0)

            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.idx_to_char[next_char_idx]

            generated += next_char
            current_seq.append(next_char_idx)

    return generated

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate text using a trained RNN model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset json file.')
    parser.add_argument('--seed_text', type=str, default='', help='Initial text to seed the generation.')
    parser.add_argument('--length', type=int, default=100, help='Length of the generated text.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (higher = more random).')  
    args = parser.parse_args()

    

    #load model and dataset
    dataset = TextDataset.load_dataset_from_json(args.dataset)
    model = TextRNN.load_model(args.model, vocab_size=dataset.vocab_size, num_layers=3, hidden_size=128)
    
    generated_text = generate_text(model, dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                               seed_text=args.seed_text, length=args.length, temperature=args.temperature) 
    print(f"Generated Text: {generated_text}")



