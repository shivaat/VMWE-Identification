README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Spanish, edition 1.1.


Corpora
-------
All annotated data come from one of these sources:
1. `Ancora`: The original Ancora corpus (see reference below).
2. `Ancora_UD`: The Universal Dependencies version of the Ancora Corpus.
3. `IXA_UD`: A corpus compiled by the IXA group in the University of the Basque country and processed with the UD tools.
4. `UD`: 3,000 sentences from the training data of the Universal Dependencies v2.0 treebank for Spanish.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated.
* UPOS (column 4): Available. Automatically annotated.
* FEATS (column 6): Available. Automatically annotated.
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated.
* MISC (column 10): No-space information available for Ancora (automatically annotated), unavailable for the UD corpora.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: IAV, IRV, LVC.full, LVC.cause, MVC, VID.


Tokenization
------------
  * Hyphens: Always split as a single token in UD.
  * Contractions: In the Ancora corpus contractions are kept as a single unit (not-split). In the UD corpora, most of them are split.


Authors
----------
The VMWEs annotations were performed by the following annotators (in alphabetical order): Cristina Aceta, Héctor Martínez Alonso, Carla Parra Escartín.


License
----------
The VMWEs annotations are distributed under the terms of the [CC-BY v4](https://creativecommons.org/licenses/by/4.0/) license.
The lemmas, POS and morphological features, contained in CONNL-U files are distributed under the terms of:
  * the [CC-BY v4](https://creativecommons.org/licenses/by/4.0/) license for the IXA corpus,
  * the GNU General Public License v.3 ([GNU GPL v.3](https://www.gnu.org/licenses/gpl.html)) for the [Ancora](http://universaldependencies.org/#es_ancora) corpus.
  * the [CC-BY-NC-SA 3.0 US](http://creativecommons.org/licenses/by-nc-sa/3.0/us/) license for the UD corpus.


Contact
----------
carla.parra@adaptcentre.ie


Reference:
----------
Mariona Taulé, Aina Peris and Horacio Rodríguez (2016) [Iarg-AnCora: Spanish corpus annotated with implicit arguments](http://dx.doi.org/10.1007/s10579-015-9334-3), in Language Resources and Evaluation 50(3), pp. 549--584.
