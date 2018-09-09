README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Hebrew, edition 1.1.


Corpora
-------
All the annotated data come from one of these sources:
1. News and articles from the Arutz 7 news website, 2001-2006, collected by the MILA Knowledge Center for Processing Hebrew (http://www.mila.cs.technion.ac.il/)
2. News and articles from the HaAretz news website, 1990-1991, collected by the MILA Knowledge Center for Processing Hebrew (http://www.mila.cs.technion.ac.il/)

The present data result from an update and an extension of the Hebrew part of the [PARSEME 1.0 corpus](http://hdl.handle.net/11372/LRT-2282), based on the source corpora above.
They extend and modify this previous version by:
* Updating the existing VMWE annotations to comply with PARSEME [guidelines eidtion 1.1](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/).
* Adding new annotated files (HaAretz corpus).
 

Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated (UDPipe).
* UPOS (column 4): Available. Automatically annotated (UDPipe).
* XPOS (column 5): Available. Automatically annotated (UDPipe).
* FEATS (column 6): Available. Automatically annotated (UDPipe).
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated (UDPipe).
* DEPS (column 8): Available. Automatically annotated (UDPipe).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: LVC.full, LVC.cause, VID. 

The UDPipe annotation relied on the model `hebrew-ud-2.0-170801.udpipe`.


Tokenization
------------
The data is tokenized by the generic corpus tokenizer, which can be downloaded from the PARSEME shared task web page.


Authors
-------
Language Leader: Chaya Liebeskind (contact: liebchaya@gmail.com)
The annotation team consists of 2 members:  Hevi Elyovich, Ruth Malka.


License
-------
The data are distributed under the terms of the [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


Citation information
--------------------
Please refer to the following 2 publications while using the Hebrew dataset:

@InProceedings{Liebeskind:2016,
  author = {Chaya Liebeskind and Yaakov HaCohen-Kerner},
  title = {A Lexical Resource of Hebrew Verb-Noun Multi-Word Expressions},
  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation},
  series  = {LREC'16},
  pages = {522--527},
  year = {2016},
  month = {may},
  date = {23-28},
  address = {Portoroz, Slovenia},
  publisher = {European Language Resources Association (ELRA)},
 }

@article{hebrew-resources:2008,
  author = {Itai, Alon and Wintner, Shuly},
  journal = {Language Resources and Evaluation},
  month = {March},
  number = {1},
  pages = {75-98},
  title = {Language resources for {H}ebrew},
  volume = {42},
  year = {2008}
}
