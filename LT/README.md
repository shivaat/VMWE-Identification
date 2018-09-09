README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Lithuanian, edition 1.1.


Corpora
-------
All annotated data come from one source:
DELFI - Lithuanian news portal http://www.delfi.lt/. Texts are published during one month period (2016-08-01 - 2016-09-01) and belong to 9 topics: car review, lifestyle, science, people, news, projects, sport, business, various.
Original article texts are merged into one file. 
There are slight differencies between version 1.0 corpus and version 1.1. corpus:
- one article was removed in version 1.1 corpus because of incorrect hyphenation;
- several sentences of another article were removed in version 1.1 corpus because of incorrect character encoding.

Total number of sentences: 11104.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated.
* UPOS (column 4): Available. Automatically annotated.
* FEATS (column 6): Available. Automatically annotated.
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause.

Automatic annotation was performed using `Semantika.lt` web service and converted to UD tagset.
MWEs of grammatical nature (multi-word pronouns, multi-word adverbs, multi-word conjunctions, multi-word-particles) which are treated as one token (with space inside) were split into parts: the MWE part of speech was assigned to each part of MWE, grammatical information was assigned to the first part only, the XPOS field was assigned the value SEQ for each non-initial part.

VMWEs in this language for version 1.0 have been annotated by two annotators, and for version 1.1 they have been reannotated by a single annotator.


Tokenization
------------
* URLs: are not recognized and might be split in parts.
* Numbers: float numbers are preserved as single tokens, unless there are spaces in the middle of the number.
* Abbreviations: dots are tokenized apart from words, e.g., prof. is tokenized as two tokens "prof" and ".".
* Each orthographic word separated by spaces is considered as a single token.
* Hybrid numerical abbreviations (with the number in digits and Lithuanian endings), e.g., 15-os (penkiolikos), 6-ąją (šeštąją), which are divided in three tokens (e.g., 15, -, os) by tokenizer, were manually corrected as one token. 
* Word forms with apostroph, e.g., men's, Eugene'as, which are divided in three tokens (e.g., Eugene, -, as) by tokenizer, were manually corrected as one token.


Authors
----------
The VMWEs annotations were performed by Jolanta Kovalevskaitė, Erika Rimkutė.
The corpus data were prepared by Loic Boizou, Ieva Bumbulienė.


License
----------
The data are distributed under the terms of the [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


Contact
----------
Jolanta Kovalevskaitė: jolanta.kovalevskaite@vdu.lt
Loic Boizou: lboizou@gmail.com
