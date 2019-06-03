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

