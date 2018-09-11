README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Hindi, edition 1.1.


Corpora
-------
All annotated data come from the Test section of the [Hindi Treebank](http://ltrc.iiit.ac.in/treebank_H2014).


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated (UDPipe).
* UPOS (column 4): Available. Automatically annotated (UDPipe).
* FEATS (column 6): Available. Automatically annotated (UDPipe).
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated (UDPipe).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: LVC.cause, LVC.full, MVC, VID.

The UDPipe annotation relied on the model `hindi-ud-1.2-160523.udpipe`.


Tokenization
------------
* Hyphens: Split as a single token.


Licence
-------
The full dataset is licensed under **Creative Commons Non-Commercial Share-Alike 4.0** licence [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)


Authors
-------
Archna Bhatia.


Contact
-------
abhatia@ihmc.us
