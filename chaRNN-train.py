import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse

from textrnn import TextRNN
from textdataset import TextDataset
    
def train_model(text_file, epochs=500, batch_size=50, learning_rate=0.001, 
                hidden_size=128, num_layers=3, seq_length=120, save_path=None):
    
    #device initialization - add support for opencl or apple silicon?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device initalialized: {device}')

    dataset = TextDataset(text_file, seq_length)
    print(f'Vocab size: {dataset.vocab_size}')
    print(f'Text length: {len(dataset.text)}')

    model = TextRNN(dataset.vocab_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if save_path:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f'Model loaded from {save_path}')

    print('Starting training...')

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        num_batches = 20

        for batch in range(num_batches):
            inputs, targets = dataset.get_batch(batch_size)
        
            inputs, targets = inputs.to(device), targets.to(device)

            # clearing the gradients for each batch - if they accumulated, we'd end up
            # with huge gradients, instability, and (probably) massive loss. 
            optimizer.zero_grad()

            # run the model on the inputs, returning the predictions
            outputs, _ = model(inputs)

            targets = targets.reshape(-1)

            # calculating the loss (uses nn.CrossEntropyLoss)
            loss = criterion(outputs, targets)

            loss.backward()  # backpropagation to calculate gradients

            #uncomment the next line for gradient clipping (for better stability)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() # update the model parameters
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        
        #DEBUG: check that the weights are being updated
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(f'Layer: {name}, Weight: {param.data}')

        prev_loss = avg_loss if epoch == 0 else prev_loss

        #print the avg loss every 10 epochs
        if (epoch +1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
            
        #if the loss increases substantially, stop training.
        #occaisional small loss fluctuations are normal, but if the loss increases
        #by more than 0.2, it may indicate that the model is diverging
        #or that the learning rate is too high.
        if avg_loss - prev_loss > .2:
            print(f'Substantial increase in loss detected at epoch {epoch + 1}. Stopping training.')
            break
        
        #save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'text_rnn_epoch_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')
    
    print('Training complete.')

    return model, dataset

if __name__ == "__main__":

    # Command line arguments: the man
    parser = argparse.ArgumentParser(description='Train an RNN model on a text file.')
    parser.add_argument('--text_file', type=str, help='Path to the training data file.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the RNN.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the RNN.')
    parser.add_argument('--seq_length', type=int, default=120, help='Sequence length ')
    parser.add_argument('--save_path', type=str, default="", help='Path to save the trained model.')

    args = parser.parse_args()

    model, dataset = train_model(args.text_file, args.epochs, args.batch_size, args.learning_rate,
                args.hidden_size, args.num_layers, args.seq_length, args.save_path)
    
    if args.save_path:
        model.export_model(args.save_path)
        dataset.export_dataset_as_json(args.save_path + '_dataset.json')
        print(f'Model and dataset exported to {args.save_path} and {args.save_path}_dataset.json')
    else:
        model.export_model('text_rnn_model.pth')
        dataset.export_dataset_as_json('text_dataset.json')
        print('Model and dataset exported to text_rnn_model.pth and text_dataset.json')
    
    print('Training complete. Model and dataset saved.')





    
