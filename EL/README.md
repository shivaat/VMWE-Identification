README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Greek, edition 1.1.


Corpora
-------
All annotated data come from one of these sources:
1. online sources:
  * kathimerini newspaper -- [online version](http://www.kathimerini.gr)
  * tovima newspaper -- [online version](http://www.tovima.gr)
  * tanea newspaper -- [online version](http://www.tanea.gr)
  * avgi newspaper -- [online version](http://avgi.gr)
  * protothema newspaper -- [online version](http://protothema.gr)
  * in.gr [newsportal](http://www.in.gr)
  * iefimerida.gr
  * efsyn.gr
  * protagon.gr
  * capital.gr
  * newsit.gr
  * espresso.gr
  * wikipedia

2. `UD`: the training data of the Universal Dependencies treebank for the Greek language.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated.
* UPOS (column 4): Available. Automatically annotated (UDPipe).
* XPOS (column 5): Available. Automatically annotated (UDPipe).
* FEATS (column 6): Available. Automatically annotated (UDPipe).
* HEAD (column 7): Available. Automatically annotated (UDPipe).
* DEPREL (column 8): Available. Automatically annotated (UDPipe).
* DEPS (column 9): Available. Automatically annotated (UDPipe).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, VPC.full. 

The UDPipe annotation relied on the model `greek-ud-2.0-170801`.


Tokenization
------------
* Contractions: Most contractions are kept as a single unit (not-split).  In the UD corpus, the forms _στου_ (_στης_, _στον_, _στη_, _στην_, _στο_, _στων_, _στους_, _στις_, _στα_) are split as two tokens _σ_ and _του_ (_της_, _τον_, _τη_, _την_, _το_, _των_, _τους_, _τις_, _τα_).


Licence
-------
All the data are distributed under the terms of the [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


Authors
-------
Voula Giouli, Vassiliki Foufi, Aggeliki Fotopoulou, Stella Markantonatou, Stella Papadelli.


Contact
-------
voula@ilsp.athena-innovation.gr
