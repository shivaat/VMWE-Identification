# VMWE-Identification

Code and documentation for the system SHOMA participated in Parseme 2018 shared task on [`automatic identification of verbal multiword expressions - edition 1.1`](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task)

We developed a ConvNet + LSTM (+ CRF) neural network architecture which recieves pre-trained embedding for words and one-hot representation for POS tags as inputs.

The data is annotated by Parseme members and more information about it is available at a [Parseme dedicated page](http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/). We converted the data to our favourable format of .pkl lists which ignore dependency parsed information which we do not use for this system.
