## Intent Detection

This repository contains the code for the master's thesis "Intent Detection Module for a Conversational Assistant".

## Installation

Clone the repository and install dependencies
```
cd intent_recognition
pipenv shell --python 3.8
pipenv install --dev
```

Build the package
```
pip3 install --editable .
```

## Torchserve
Start the TorchServe server (you might want to do it in another window)
```
mkdir model_store
torchserve --start --model-store model_store
```

To stop the server press CTRL+C and run
```
mkdir model_store
torchserve --stop
```

## The intent detection module
If the server is running, start the module using the command:
```
python3 intent_detection/main.py
```

Note that if you run the program for the first time, the embedding model must pre-compute embeddings, which might take some time.
