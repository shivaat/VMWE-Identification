README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Turkish, edition 1.1.


Corpora
-------
All annotated data come from Turkish newspaper sources. The Turkish dataset is annotated according to the PARSEME Shared Task on Automatic Identification of Verbal Multiword Expressions 1.1 Guidelines.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated (ITU NLP pipeline).
* UPOS (column 4): Available. Automatically annotated (ITU NLP pipeline) and mapped to Universal POS tags.
* FEATS (column 6): Available. Automatically annotated (ITU NLP pipeline) and mapped to Universal POS tags.
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated (ITU NLP pipeline) and mapped to Universal Dependencies.
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: LVC.full, VID, MVC.

The Turkish annotated corpus consists of 18611 sentences in total, with the following VMWEs: 3450 LVC.full, 3677 VID, 2 MVC.

ITU NLP pipeline:
http://tools.nlp.itu.edu.tr/
\cite{itunlp}


Tokenization 
------------
The inflectional group (IG) formalism  has become the de facto standard in parsing Turkish. According to the formalism, orthographic tokens are divided into morphosyntactic words from derivational boundaries (\cite{udturkish}).


Authors
-------
Language Leader: Tunga Güngör (contact: gungort@boun.edu.tr)
The annotation team consists of 2 members: Berna Erden, Gözde Berk.


Copyright information
---------------------
Creative Commons  CC-BY-NC-SA License.
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode


Citation information
--------------------
(in review)
Please refer to the following publication while using the Turkish dataset:
@InProceedings{turkishdataset,
  author = {Berna Erden, Gozde Berk, and Tunga Gungor},
  title = {Turkish Verbal Multiword Expressions Corpus},
  booktitle = {26th IEEE Signal Processing and Communications Applications Conference, SIU 2018},
  month = {May},
  year = {2018},
  address = {İzmir, Turkey},
  pages={}
}


Licence
-------
The full dataset is licensed under Creative Commons Non-Commercial Share-Alike 4.0 licence CC-BY-NC-SA 4.0


References
----------
@InProceedings{itunlp,
  author = {Eryigit, Gulsen},
  title = {ITU Turkish NLP Web Service},
  booktitle = {Proceedings of the Demonstrations at the 14th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  month = {April},
  year = {2014},
  address = {Gothenburg, Sweden},
  publisher = {Association for Computational Linguistics},
}

@InProceedings{udturkish,
  author    = {Sulubacak, Umut  and  Gokirmak, Memduh  and  Tyers, Francis  and  Coltekin, Cagri  and  Nivre, Joakim  and  Eryigit, Gulsen},
  title     = {Universal Dependencies for Turkish},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  publisher = {The COLING 2016 Organizing Committee},
  pages     = {3444--3454},
  url       = {http://aclweb.org/anthology/C16-1325}
}
