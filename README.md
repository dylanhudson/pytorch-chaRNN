# pytorch-chaRNN

## Background
I'm trying to get back into learning PyTorch, so I figured I'd port this classic project based on <https://github.com/karpathy/char-rnn> to run in a more modern environment, using <https://github.com/nikhilbarhate99/Char-RNN-PyTorch> as a guide.  


# Setup and Environtment
I have this running in Python 3.11.9 on MacOS 14.7.6 (x86) with torch 2.1.1. Torch isn't yet compatible with more recent versions of numpy, so I had to downgrade numpy to 1.26.4. 
If you're new to Python, I suggest setting up a virtual environment, especially for using PyTorch, as things can get hairy. You can find more details [here](https://docs.python.org/3/library/venv.html), but here's the quickstart version: \
Open a new terminal window, navigate to where you want your working directory, and create a folder for the project. Enter the folder, and run\
`python3 -m venv .venv`\
`source .venv/bin/activate`\
Then proceed to install PyTorch, etc. If you open a new terminal window, you will need to again activate the virtual env with `source` command above.


# Usage
## Jupyter Notebook
A notebook is provided in jupyter/pytorch-chaRNN.ipynb. I ran it with jupyter_client 8.6.3 and jupyter_core 5.8.1.

## Command-line
To train a model, run chaRNN-train.py with your text corpus as the command-line argument. \
To generate text with your model, use the generate-text.py script, with --model "path/to/model.pth" --length \[number of chars to generate], --temp \[temperature value \(float)], and --seed "my seed text". \
Example: `python3 generate-text.py --model "shakespeare-model.pth" --length 30 --temp 1.2 --seed "Alas poor Yorick"`

## Datasets
I've included the Shakespeare dataset here in /datasets so you too may generate beautiful works like this emotional segment of the well-known *"Much Alief About Gandeth"*:\

>JULIET:\
>Go, go alief feech it, right.
>
>ISABELLA:\
>A gandeth.
>
>BRUTUS:\
>What courteanly grav.  


Then, when anyone asks you "What's the big deal about all this GenAI stuff?" you can demonstrate the immense utility and beauty of the recurrent neural network.  











