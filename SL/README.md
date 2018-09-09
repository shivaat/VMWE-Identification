README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Slovene, edition 1.1.

Corpora
-------
All annotated data come from ssj500k training corpus, which is available and described here:
* CLARIN.SI repository: http://hdl.handle.net/11356/1181
* Description: http://eng.slovenscina.eu/tehnologije/ucni-korpus
The present data extend and modify the previous corpora by:
* adding the verbal multiword expression annotation layer, according to the PARSEME Shared Task 1.1 [guidelines](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/).
* adding the automatically generated layer of syntactic dependencies (see below for details)

Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:
* LEMMA (column 3): Available (manually annotated).
* XPOS (column 5): Available (manually annotated). The tagset is [JOS](http://nl.ijs.si/jos/josMSD-en.html).
* (column 6): NOT available (Morphological features are encoded in the tag after the first letter representing POS).
* HEAD (column 7): Head of dependency relations available (manually annotated). The system is described in [JOS/SSJ](http://eng.slovenscina.eu/tehnologije/razclenjevalnik).
* DEPREL (column 8): Dependecy relations available (manually annotated). The inventory is described in [JOS/SSJ](http://eng.slovenscina.eu/tehnologije/razclenjevalnik). Only the first 11411 sentences contain DEPREL annotations, while the other 2100 do not. 
* DEPS (column 9): Not available.
* MISC (column 10): No-space information available.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.cause, LVC.full, IAV, IRV.

Sentences from 1 to 8641 have been annotated by two annotators per file. Sentences from 8642 to 11411 have been annotated by one annotator per file. In sentences 1 to 11411, the MWE categories from the first version of the annotation guidelines (i.e. ID, LVC, VPC, IReflV, OTH) were first automatically converted to the new categories (ID -> VID, LVC -> LVC.cause/LVC.full, VPC -> IAV, IReflV -> IRV, OTH -> various categories). Next, the sentences were manually checked by two annotators in order to remove inconsistencies. An additional 2100 sentences were annotated (5 packages of 400 sentences by a single annotator and 1 package of 100 sentences by all four annotators).

Tokenization
------------
* Language-specific tokenization rules are applied. Tokenization is manually checked in the corpus.

Annotation
----------


Authors
----------
All VMWEs annotations were performed by Polona Gantar, Taja Kuzman, Teja Kavčič, Špela Arhar Holdt and Simon Krek.

License
----------
The data are distributed under the terms of the [Creative Commons BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Contact
----------
simon.krek@ijs.si
