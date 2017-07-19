# NLPiano

Automatic music generation using Magenta and recent NLP techniques.

## Requirements

The project requires Tensorflow and Magenta to build.

classical_training_data.zip contains all training data used, which can be found [here](http://www.piano-midi.de/). To run on Ubuntu,

```
sudo apt-get install unzip
unzip classical_training_data.zip -d data/
python main.py
```