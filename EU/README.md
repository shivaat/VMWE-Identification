README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Basque, edition 1.1.


Corpora
-------
All annotated data come from one of these sources (subcorpora):
1. `UD`: 6621 sentences, the whole Universal Dependencies treebank for Basque.
2. `Elhuyar`: 4537 sentences taken from the [Elhuyar Web Corpora](https://labur.eus/rC75P).


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated.
* UPOS (column 4): Available. Automatically annotated for Elhuyar, manually annotated for UD.
* FEATS (column 6): Available. Automatically annotated.
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated for Elhuyar, manually annotated for UD. The inventory is [Universal Dependency Relations](http://universaldependencies.org/u/dep).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated by a single annotator per file. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full and LVC.cause.

Lemmas and morphological features for the UD corpus were automatically annotated by [Eustagger](http://ixa.eus/node/4450). POS-tags and dependency relations in the UD corpus were first annotated based on a Basque tagset and then automatically converted to UD tags. The Elhuyar corpus was entirely annotated by the [Mate parser](https://code.google.com/archive/p/mate-tools/).


Known issues
------------
* The Elhuyar subcorpus consists of texts which were automatically extracted from the web. Although only good-quality sources were selected, a few strange sentences can be found in the corpus due to automatic extraction (like sentences missing some words).
* Lemmas and POS-tags in the Elhuyar corpus are not always reliable. This can cause some trouble concerning a few LVCs where the noun is erroneously tagged as a verb.


Licence
-------
The full dataset is licensed under **Creative Commons Non-Commercial Share-Alike 4.0** [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)


Authors
-------
(listed in alphabetical order) Itziar Aduriz, Ainara Estarrona, Itziar Gonzalez-Dios, Antton Gurrutxaga, Uxoa Iñurrieta and Ruben Urizar.


Contact
-------
{usoa.inurrieta}@ehu.eus
