## Intent Detection

This repository contains the code for the master's thesis "Intent Detection Module for a Conversational Assistant".

## Installation

Install dependencies
```
cd intent_recognition
pipenv shell --python 3.8
pipenv install --dev
```

Build the package
```
pip3 install --editable .
```

Start the TorchServe server (you might want to do it in another window)
```
torchserve --start --model-store model_store
```

Run the module
```
python3 intent_detection/main.py
```

Note that if you run the program for the first time, the embedding model must pre-compute embeddings, which might take some time.