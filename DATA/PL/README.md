README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Polish, edition 1.1.

Corpora
-------
All the annotated data come from one of these sources:
1. `PCC`: [Polish Coreference Corpus](http://zil.ipipan.waw.pl/PolishCoreferenceCorpus), the 21 "long" texts from this corpus are included, 36,000 tokens, Rzeczpospolita newspaper.
2. `NKJP`: [National Corpus of Polish](http://clip.ipipan.waw.pl/NationalCorpusOfPolish), all texts from daily newspapers and magazines are included, i.e. those whose identifiers start with 130-2, 130-3, 130-5, 120-, 310- and 330-.
Part of the NKJP corpus was manually annotated for syntax and released as [Składnica](http://zil.ipipan.waw.pl/Sk%C5%82adnica) treebank. It was then converted into UD format by Alina Wróblewska and released as the [Polish Dependency Treebank](http://zil.ipipan.waw.pl/PDB/). Sentences with identifiers starting with 120-, 310- and 330- stem from this last corpus.

The present data result from an update and an extension of the Polish part of the [PARSEME 1.0 corpus](http://hdl.handle.net/11372/LRT-2282), based on the source corpora above.
They extend and modify this previous version by:
* updating the existing VMWE annotations to comply with PARSEME [guidelines eidtion 1.1](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/).
* adding new annotated files (with sentences having identifiers starting with 120-, 310- and 330-)

Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Manually double-annotated and adjudicated for the NKJP files, automatically annotated for the PCC files.
* UPOS (column 4): Available. Automatically converted from the manually annotated NKJP tagset, with the [Polish UD conversion table](http://universaldependencies.org/docs/tagset-conversion/pl-ipipan-uposf.html). The tagset is the [Universal POS-tags](http://universaldependencies.org/u/pos).
* XPOS (column 5): Available. Manually double-annotated and adjudicated for the NKJP files, automatically annotated for the PCC files. The [NKJP tagset](http://nkjp.pl/poliqarp/help/ense2.html) is used .
* FEATS (column 6): Available. Automatically converted from the manually annotated NKJP tagset, with the [Polish UD conversion table](http://universaldependencies.org/docs/tagset-conversion/pl-ipipan-uposf.html) extended with some missing categories and feature combinations. The [UD tagset](http://universaldependencies.org/u/feat/index.html) is used.
* HEAD (column 7): Available. Automatically converted from the manual annotation in [Składnica](http://zil.ipipan.waw.pl/Sk%C5%82adnica), for sentences with identifiers starting with 120-, 310- and 330-. Automatically annotated by [UDPipe](https://ufal.mff.cuni.cz/udpipe) with the Polish [model version 1.2-160523](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1659). 
* DEPREL (column 8): Available. Automatically converted from the manual annotation in [Składnica](http://zil.ipipan.waw.pl/Sk%C5%82adnica), for sentences with identifiers starting with 120-, 310- and 330-. Automatically annotated by [UDPipe](https://ufal.mff.cuni.cz/udpipe) with the Polish [model version 1.2-160523](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1659). The tagset is [Universal Dependency Relations](http://universaldependencies.org/u/dep).
* DEPS (column 9): Available for sentences with identifiers starting with `120-2-*`, `310-*` and `330-*` only.
* MISC (column 10): No-space information available. Manually double-annotated and adjudicated for the NKJP files, automatically annotated for the PCC files.
* PARSEME:MWE (column 11): Manually annotated by a single annotator per file. The following [categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are used: IAV, IRV, LVC.full, LVC.semi, VID. 

Tokenization
------------
Manually double-annotated and adjudicated for the NKJP files, automatically annotated for the PCC files.

Authors
----------
All VMWEs annotations (column 11) were performed by Agata Savary. For authorship of the data in columns 1-10 see the original corpora.

License
----------
The VMWEs annotations (column 11) are distributed under the terms of the [CC-BY v4](https://creativecommons.org/licenses/by/4.0/) license.
The lemmas, POS-tags, morphological and features (columns 1-6), are distributed under the terms of the [CC-BY-SA 0.4](https://creativecommons.org/licenses/by-sa/4.0/) license for the sentences with identifiers starting with 120-2-*, 310-* and 330-*, and under the terms of the ([GNU GPL v.3](https://www.gnu.org/licenses/gpl.html)) for other sentences. Dependency relations (columns 7-9)
are distributed under the terms of the [CC-BY-SA 0.4](https://creativecommons.org/licenses/by-sa/4.0/) license.

Contact
----------
agata.savary@univ-tours.fr

