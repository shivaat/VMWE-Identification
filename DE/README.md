README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for German, edition 1.1.


Corpora
-------
The annotated training data come from the annual workshop on statistical
machine translation, [WMT 2015](http://statmt.org/wmt15/translation-task.html#download)
news2013: News Crawl: articles from 2013 (Bojar et al. 2016).

The annotated test data are sentences 1500--3000 from the [German part](https://github.com/UniversalDependencies/UD_German-GSD) of the [Universal Dependencies (UD) corpus](http://universaldependencies.org).


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated (UDPipe).
* UPOS (column 4): Available. Automatically annotated for the training data (UDPipe), manually annotated for the test data.
* XPOS (column 5): Available. Automatically annotated for the training data (UDPipe), automatically annotated for the test data (TreeTagger).
* FEATS (column 6): Available. Automatically annotated (UDPipe).
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated for the training data (UDPipe), manually annotated for the test data. The inventory is [Universal Dependency Relations](http://universaldependencies.org/u/dep).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: IRV, LVC.cause, LVC.full, VID, VPC.full, VPC.semi.


Tokenization
------------
Training data: The tokenization was performed using the WMT script for German tokenization.


Annotation
----------


Authors
----------
Building on the annotations made by Fabienne Cap and Glorianna Jagfeld in the previous version (v1.0), VMWE annotations were performed by Timm Lichte and Rafael Ehren.


License
----------
All VMWEs annotations are distributed under the terms of the [CC-BY v4](https://creativecommons.org/licenses/by/4.0/) license.
The lemmas, POS-tags, morphological features and dependency tags (contained in the CoNLL-U files) are distributed under the [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/) license, i.e. the same license as the [German Universal Dependencies data](http://universaldependencies.org/#de), on which [UDPipe](https://ufal.mff.cuni.cz/udpipe) was trained.


Contact
----------
lichte@phil.hhu.de


References
----------
Bojar, Ondrej, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurelie Neveol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Carolina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor & Marcos Zampieri. 2016. Findings of the 2016 Conference on Machine Translation (WMT16). In Proceedings of the First Conference on Machine Translation (WMT16), Volume 2: Shared Task Papers, 131--198. 
