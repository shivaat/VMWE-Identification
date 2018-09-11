README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for French, edition 1.1.


Corpora
-------
The verbal MWEs have been annotated in the following corpora:
1. `sequoia`: all the 3099 sentences of the [Sequoia Treebank](https://www.rocq.inria.fr/alpage-wiki/tiki-index.php?page=CorpusSequoia)
2. `fr-ud`: the 2.1 version of the French universal dependencies treebank (recently renamed "GDS" for Google dataset)
3. `fr_partut-ud`: the 2.1 UD version of the French part of the ParTUT
4. `fr_pud-ud`: the first 500 sentences of the French part of the 2.1 UD version of the Parallel Universal Dependencies (PUD) treebanks created for the [CoNLL 2017 shared task on Multilingual Parsing from Raw Text to Universal Dependencies](http://universaldependencies.org/conll17/).


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available.
* UPOS (column 4): Available. Manually annotated.
* HEAD and DEPREL (columns 7 and 8): Available. Manually annotated. The inventory is [Universal Dependency Relations](http://universaldependencies.org/u/dep)
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, IRV, MVC.

The CoNLL-U columns are those found in the UD 2.1 release (for the Sequoia corpus, the UD 2.1 version results from an automatic conversion by Bruno Guillaume).
So the annotation scheme for POS tags and syntactic dependencies are relatively homogeneous.
Note though that differences remain, as the UD guidelines may have been interpreted differently by the various teams having produced the different corpus.


Tokenization
------------
* The tokenization is that of the French UD treebanks, in which the following contractions appear as multi-word tokens (e.g. 1-2 au), split into words:
E.g. : Au soleil
```
1-2 Au
1 à
2 le
3 soleil
```

The list of contractions is:
```
au
auquel
aux
auxquelles
auxquels
des
desquelles
du
duquel
```

Note that the only ambiguous case are "des" / "du". Depending on the context, these tokens are either a plain determiner, or are split into preposition "de" + determiner "le" / "les".


Authors
----------
The VMWEs annotations were performed by Marie Candito, Mathieu Constant, Caroline Pasquer, Yannick Parmentier, Carlos Ramisch, Jean-Yves Antoine.
The annotations for the new test set for the 1.1 shared task were performed by Marie Candito.


Licence
----------
The VMEs annotations are distributed under the terms of the [CC-BY v4 license](https://creativecommons.org/licenses/by/4.0/). As far as the CONLL-U files are concerned, the UD part of the corpus is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the Sequoia part is under [LGPL-LR](http://infolingu.univ-mlv.fr/DonneesLinguistiques/Lexiques-Grammaires/lgpllr.html). UD sentences can be identified by their `sentid` prefixed with `fr-ud`.


Contact
----------
`marie.candito@gmail.com`
