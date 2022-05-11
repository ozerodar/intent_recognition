## Intent Detection

This repository contains the code for the master's thesis "Intent Detection Module for a Conversational Assistant".

## Requirements

The module requires Python 3.8, pip3 and pipenv installed on your machine.

## Installation

Clone the repository and install dependencies.
```
cd intent_recognition
pipenv shell --python 3.8
pipenv install --dev
```

Build the package
```
pip3 install --editable .
```

Start the TorchServe server. You might want to do it in another window (within the same environment).
```
mkdir model_store
torchserve --start --model-store model_store
```

Once the server is running, start the module using the command:
```
python3 intent_detection/main.py
```

Note that if you run the program for the first time, the embedding model must pre-compute embeddings, and it might take several minutes.
The application is ready to after the message "Application startup complete."

The Uvicorn server will run on: http://0.0.0.0:5555

## Stopping the module

To stop the server press CTRL+C and run
```
torchserve --stop
```
