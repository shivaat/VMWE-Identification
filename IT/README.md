README
------
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Italian, edition 1.1.

Corpora
-------
All annotated data comes from the [PAISÀ Corpus](http://www.corpusitaliano.it/en/) converted to CoNLL-U format (added two additional fields with underscores). The original corpus is distributed under Creative Commons license, Attribution-ShareAlike and Attribution-Noncommercial-ShareAlike.

This corpus includes subcorpora with sentences from it_blog, it_wikinews and it_wikipedia.

Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated. 
* UPOS (column 4): Available. Automatically annotated. See [Corpus description](http://www.corpusitaliano.it/en/contents/description.html).
* FEATS (column 6): Available. Automatically annotated. See [Corpus description](http://www.corpusitaliano.it/en/contents/description.html)
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated. See [Corpus description](http://www.corpusitaliano.it/en/contents/description.html)
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. VMWEs in this language have been annotated by a single annotator per file. The following categories are used (http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/):  LVC.full, LVC.cause, VID, VPC.full, VPC.semi, IRV, IAV, MVC, LS.ICV. 

Tokenization
-------------
The tokenization follows the original tokenization of the PAISÀ corpus with the exception of compound prepositions 

Some pre-processing has been applied to the original files of the corpus in order to split compound prepositions (dei, nei, delle, etc.).
To this end we added new tokens corresponding to the components of the compound prepositions (see example below)  and we also realigned all the dependency index: the heuristic being used is that the preposition is the head of the prepositional article (all tokens pointing to the prepositional article will point to the preposition in the split version and the determiner also points to the preposition).

For instance the original CONLL-U sentence:

|Rank|Surf|Lemma|PosG|PosF|Morph|DepIndex|DepLabel|x|x|
|----|----|-----|----|----|-----|--------|--------|-|-|
| 1 | Perchè | Perchè | C | CS | _ | 4 | mod | _ | _ |
| 2 | la | il | R | RD | num=s\|gen=f | 3 | det | _ | _ |
| 3 | ragione | ragione | S | S | num=s\|gen=f | 4 | subj | _ | _ |
| 4 | sta | stare | V | V | num=s\|per=3\|mod=i\|ten=p | 0 | ROOT | _ | _ |
| 5 | nel | in | E | EA | num=s\|gen=m | 4 | comp | _ | _ |
| 6 | mezzo | mezzo | S | S | num=s\|gen=m | 5 | prep | _ | _ |
| 7 | no | no | B | BN | _ | 4 | neg | _ | _ |
| 8 | ? | ? | F | FS | _ | 4 | punc | _ | _ |

Is transformed into the following:

|Rank|Surf|Lemma|PosG|PosF|Morph|DepIndex|DepLabel|x|x|
|----|----|-----|----|----|-----|--------|--------|-|-|
| 1 | Perchè | Perchè | C | CS | _ | 4 | mod | _ | _ |
| 2 | la | il | R | RD | num=s\|gen=f | 3 | det | _ | _ |
| 3 | ragione | ragione | S | S | num=s\|gen=f | 4 | subj | _ | _ |
| 4 | sta | stare | V | V | num=s\|per=3\|mod=i\|ten=p | 0 | ROOT | _ | _ |
| 5-6 | nel | _ | _ | _ | _ | _ | _ | _ | _ |
| 5 | in | in | E | E | _ | 4 | comp | _ | _ |
| 6 | il | il | R | RD | _ | 5 | det | _ | _ |
| 7 | mezzo | mezzo | S | S | num=s\|gen=m | 5 | prep | _ | _ |
| 8 | no | no | B | BN | _ | 4 | neg | _ | _ |
| 9 | ? | ? | F | FS | _ | 4 | punc | _ | _ |

In addition we also introduced the `SpaceAfter=No` tag on the word preceding a clitic beloging to the same token, e.g., lavar-si. These are annotated as two separate words in the the original corpus.

Licence
-------
The full dataset is licensed under Creative Commons Non-Commercial Share-Alike 4.0 licence CC-BY-NC-SA 4.0.

Authors
-------
Johanna Monti, Valeria Caruso, Maria Pia di Buono, Antonio Pascucci, Annalisa Raffone, Anna Riccio, Federico Sangati.

Contact
-------
jmonti@unior.it or federico.sangati@gmail.com
