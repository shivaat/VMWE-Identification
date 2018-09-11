# VMWE-Identification

Code and documentation for the system SHOMA participated in Parseme 2018 shared task on [`automatic identification of verbal multiword expressions - edition 1.1`](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task).

We developed a ConvNet + LSTM (+ CRF) neural network architecture which recieves pre-trained embedding for words and one-hot representation for POS tags as inputs.

The data is annotated by Parseme members and more information about it is available at a [Parseme dedicated page](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/). We converted the data to our favourable format of .pkl lists which ignore dependency parsed information which we do not use for this system. The labels in our .pkl lists are in IOB-like format.

## Requirements

* Python 3
* keras with a tensorflow backend
* Gensim


## Usage
The scripts for running the code are provided in the directory `CODE`. The script `test_run.py` is a sample code for training the system and testing it on selected languages which are listed in the list `languages`. 

`python3 test_run.py`

It calls `script.py` with the parameters in `langs.json`. The `langs.json` file includes the training settings for each language including the train, dev and test files, word2vec path, the name of the model you would like to run and the initial weight (if you have some saved weights from before and you want to continue on that).
The `test_run.py`, at the moment, runs the program for two languages `ES` and `EN` with a simple trained word2vec (which we uploaded in this directory just as an example). 
In order to run the program for any other languages, list their code in the list `languages`. 
For each language, please download the relevant word2vec from [fastText repository](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) and add its path to the `langs.json` file as the value for `word2vec_dir`. 
Other parameters in `test_run.py` are the number of epochs, batch size and if you want to use POS features which are set by `l.set_params(10, 100, True)`. 
