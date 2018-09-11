README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Croatian, edition 1.1.


Corpora
-------
All annotated data comes from the following source:
1. [SETimes.HR](http://nlp.ffzg.hr/resources/corpora/setimes-hr/): 4137 sentences from the Croatian part of the SETimes newspaper corpus.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated.
* UPOS (column 4): Available. Automatically annotated.
* FEATS (column 6): Available. Automatically annotated.
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated.
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, IRV, IAV.

VMWEs in this language have been annotated by a single annotator per file, with the exception of the 300 sentences in the *test* set, and the last 535 sentences of the *train* set, which were annotated by three annotators.


Tokenization
------------
* Hyphens: Hyphenated compounds preserved as a single token.


Licence
-------
The full dataset is licensed under **Creative Commons Non-Commercial Share-Alike 4.0** licence [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)


Authors
-------
Goranka Blagus, Maja Buljan, Nikola Ljubešić, Ivana Matas, Jan Šnajder


Contact
-------
maja.buljan@ims.uni-stuttgart.de
nljubesi@ffzg.hr
jan.snajder@fer.hr
