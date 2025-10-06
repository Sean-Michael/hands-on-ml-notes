# Hands-On Machine Learning with Scikit-Learn and PyTorch Notes

This repository will store my notes, example code, and anything else related to my progression through this book. 

## Chapters

Each chapter will have it's own directory. Section notes, and example code will be included in these folders. The notes may be split between many files or contained in a single `NOTE.md`.

## Notes format

Each note file will be a markdown document that has formatted subheadings, lists, code blocks, and other readable tricks to keep it interesting and useful for future reference. Sometimes example code will be referenced in the notes, or simply included in-line. 

### Questions 

When I have a Question I'll type it out in a numbered list at the bottom of the page. This way I can go back and ask Claude or Google to research the answer later, and I'll include that in an *Answer:* subheader.  

## Example Code

The author's included example repository has been cloned within this repository for easy access, and reference. [handson-mlp](./handson-mlp/)

### Installation

For local development on my Macbook I performed the following steps after cloning the example repository. [Instructions](https://github.com/ageron/handson-mlp/blob/main/INSTALL.md)

I already had the latest major version of python3 installed.

```bash
% python3 --version
Python 3.13.5
```

Install Anaconda with homebrew:

```bash
# Install Anaconda with homebrew:
brew install --cask anaconda

# Add anaconda bin dir to PATH environment variable
echo 'export PATH="/opt/homebrew/anaconda3/bin:$PATH"' >> ~/.zshrc

# Refresh terminal session for changes to take effect
source ~/.zshrc

# Initialize anaconda
conda init zsh

# Update conda to latest version (requires sudo)
conda update -n base -c defaults conda

# Change to the handson-mlp directory
cd handson-mlp

# Create a new conda environment
conda env create -f environment.yml

# And activate it 

conda activate homlp

# Register the environment to Jupyter
python3 -m ipykernel install --user --name=python3

# Then start jupyter
jupyter lab

# Or notebook
jupyter notebook
```

To make things easy for myself I created the following alias to activate the conda environment and start jupyter. 

```bash
alias homlp='cd handson-mlp && \
        /opt/homebrew/anaconda3/bin/conda activate homlp && \
        /opt/homebrew/anaconda3/bin/jupyter lab'
```