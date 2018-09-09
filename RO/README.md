README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Romanian, edition 1.1.


Corpora
-------
All annotated data comes from the "Agenda" newspaper. Some of them are part of the Romanian Universal Dependencies treebank (RoRefTrees).


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
* PARSEME:MWE (column 11): Manually annotated by a single annotator per file. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, IRV.

The UDPipe annotation relied on the model `romanian-ud-2.0-170801.udpipe`.


Tokenization
------------
The training and testing files are annotated with the TTL tool (developed at ICIA, by Radu Ion, 2007).


Authors
-------
All VMWEs annotations were performed by Verginica Barbu Mititelu, Mihaela Onofrei, Mihaela Ionescu, and Monica-Mihaela Rizea.


License
-------
The data are distributed under the terms of the CC BY v4 License.


Contact
-------
vergi@racai.ro 
