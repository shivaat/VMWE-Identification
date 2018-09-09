README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Brazilian Portuguese, edition 1.1.

Corpora
-------
All annotated data come from one of these sources (subcorpora):
1. `DG`: 19,040 sentences from the Diário Gaúcho newspaper.
2. `UD`: 9,664 sentences from the training corpus of the Universal Dependencies v2.1 treebank for Brazilian Portuguese UD_Portuguese-BR (renamed to UD_Portuguese-GSD in most recent UD releases).

Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* FORM (column 2): Available. Provided in the original corpora and manually corrected for token _en_ when in contractions: see [this issue](https://github.com/UniversalDependencies/UD_Portuguese-GSD/issues/9)
* LEMMA (column 3): Available. Automatically provided by UDPipe in both DG and UD. Some systematically wrong lemmas were manually corrected such as words ending in _ões_
* UPOS (column 4): Available. Manually annotated for UD, automatically provided by UDPipe for DG. Uses UD tagset.
* FEATS (column 6): Available. Automatically provided by UDPipe in both DG and UD. Uses UD tagset.
* HEAD and DEPREL (columns 7 and 8): Available. Manually annotated for UD, automatically provided by UDPipe for DG. Uses UD tagset.
* MISC (column 10): No-space information available. Manually annotated for UD, automatically provided by UDPipe for DG.
* PARSEME:MWE (column 11): Manually annotated by a single annotator per file. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, IRV.

The UDPipe annotation relied on the model `portuguese-ud-2.0-conll17-170315.udpipe` trained on Portuguese Bosque UD treebank.

Tokenization
------------
* We have re-tokenized DG using UDPipe's tokenizer trained on Bosque. Contractions and hyphenated clitics (including many IRV) have been improved wrt release 1.0.
* For UD, we used the reference tokenization provided in the UD2.1 release (which has some problems, as mentioned below)
* In both corpora, contractions are split, but DG may contain errors due to automatic processing

Known issues
------------
* Single-token `IRV`s: The hyphenization inconsistency between DG and UD is relevant for reflexive verbs with proclisis, where the clitic appears after the verb with a hyphen (e.g. _queixar-se_ lit. _complain-self_ 'to complain'). They do not have the same tokenization in both subcorpora. For example, _queixar-se_ is a single token annotated as `IRV` in DG, whereas it consists of three tokens _queixar - se_ in UD. In the latter case, the hyphen is **not** annotated as part of the `IRV`. We intend to fix it in future versions.
* Lemmas: The quality of the automatic lemmatizer is limited because it was learned on a small treebank and not checked using dictionaries. See [this issue](https://github.com/UniversalDependencies/UD_Portuguese-GSD/issues/8) for UD lemmas.
* Merging _ea_ and _eo_: The UD corpus strangely merges _e_+_a_ (_and_+_the.FEM_) and _e_+_o_ (_and_+_the.MASC_) at some places.  We have left the surface untouched (as "ea" and "eo"), to be able to keep the same tokenization as in the original corpus. See [this issue](https://github.com/UniversalDependencies/UD_Portuguese-GSD/issues/5) for the _eo_ problem.


License
-------
The full dataset is licensed under **Creative Commons Non-Commercial Share-Alike 4.0** licence [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Authors
-------
For version 1.1: Silvio Cordeiro, Carlos Ramisch, Renata Ramisch, Leonardo Zilio.
For version 1.0: Helena Caseli, Silvio Cordeiro, Carlos Ramisch, Renata Ramisch, Aline Villavicencio, Leonardo Zilio.

Contact
-------
Corpus processing: {carlos.ramisch,silvio.cordeiro}@lis-lab.fr
VMWE annotations: renata.ramisch@gmail.com

