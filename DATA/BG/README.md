README
======
This is the README file from the PARSEME verbal multiword expressions (VMWEs) corpus for Bulgarian, edition 1.1.


Corpora
-------
All annotated data come from the [Bulgarian National Corpus](http://dcl.bas.bg/bulnc/en/). Only text samples from the public domain are selected that are not subject to copyright.

For some text samples originality could not be confirmed and there might be some translational texts.


Provided annotations
--------------------
The data are in the [.cupt](http://multiword.sourceforge.net/cupt-format) format. Here is detailed information about some columns:

* LEMMA (column 3): Available. Automatically annotated (UDPipe).
* UPOS (column 4): Available. Automatically annotated (UDPipe).
* XPOS (column 5): Available. Automatically annotated (UDPipe).
* FEATS (column 6): Available. Automatically annotated (UDPipe).
* HEAD and DEPREL (columns 7 and 8): Available. Automatically annotated (UDPipe).
* DEPS (column 9): Available. Automatically annotated (UDPipe).
* MISC (column 10): No-space information available. Automatically annotated.
* PARSEME:MWE (column 11): Manually annotated. The following [VMWE categories](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/?page=030_Categories_of_VMWEs) are annotated: VID, LVC.full, LVC.cause, IRV, IAV (pilot annotation).

IAV have been annotated by some annotators and are thus not fully covered.

The UDPipe annotation relied on the model `bulgarian-ud-2.0-170801.udpipe`.


License
-------
The processed and annotated corpus is distributed under the license Creative Commons Attribution 4.0 International [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


Authors
-------
Ivelina Stoyanova, Svetla Koeva, Svetlozara Leseva, Maria Todorova, Tsvetana Dimitrova, Valentina Stefanova


Contacts
--------
iva@dcl.bas.bg
[http://dcl.bas.bg/](http://dcl.bas.bg/)

[Annotation notes](http://dcl.bas.bg/en/parseme-corpus/)
