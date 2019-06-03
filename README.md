# VMWE-Identification

This repository contains the code and documentation for several different neural architectures to identify [Verbal Multiword Expressions](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task).

There are three main approaches each in one directory:
1) SHOMA: a ConvNet + LSTM (+ CRF) neural network architecture that participated in Parseme 2018 shared task on [`automatic identification of verbal multiword expressions - edition 1.1`](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task).
2) MTL: Multi-task learning
3) TRL: Cross-lingual Transfer Learning


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

## Citation

    @article{DBLP:journals/corr/abs-1809-03056,
      author    = {Shiva Taslimipoor and
                   Omid Rohanian},
      title     = {{SHOMA} at Parseme Shared Task on Automatic Identification of VMWEs:
                   Neural Multiword Expression Tagging with High Generalisation},
      journal   = {CoRR},
      volume    = {abs/1809.03056},
      year      = {2018},
      url       = {http://arxiv.org/abs/1809.03056},
      archivePrefix = {arXiv},
      eprint    = {1809.03056},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1809-03056},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }

