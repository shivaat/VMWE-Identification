
import os
from corpus import Corpus
import pickle

class Corpus_reader:

	def __init__(self, path):

		parsemeCorpus = Corpus(path)
		self.train_sents = None
		self.dev_sents = None
		self.test_sents = None
		if parsemeCorpus.sentences:
                        self.train_sents = parsemeCorpus.sentences
		if parsemeCorpus.devSents:
                        self.dev_sents = parsemeCorpus.devSents
		if parsemeCorpus.testSents:
                        self.test_sents = parsemeCorpus.testSents
		#self.test_sents, self.devMweNum = parsemeCorpus.readCuptFile(path+"test.cupt")

	def read(self, sents):
		seqs = []
		for s in range(0, len(sents)):
			seqs_i = []
			active_mwe = 0
			for t in range(len(sents[s].tokens)):
				if len(sents[s].tokens[t].parentMWEs) > 0:
					tag = ''
					mwe = sents[s].tokens[t].parentMWEs[0]
					if sents[s].tokens[t] == mwe.tokens[0]: # Check if the token is the first components of MWE
						tag = tag + 'B_'
						if not mwe.isSingleWordExp:
							active_mwe = 1
					else:
						tag = tag + 'I_'
						if sents[s].tokens[t] == mwe.tokens[-1]: # Check if the token is the first components of MWE
							active_mwe = 0
					tag = tag + mwe.type
					for mwe in sents[s].tokens[t].parentMWEs[1:]:
						if sents[s].tokens[t] == mwe.tokens[0]:
							tag = tag + ';B_'
						else:
							tag = tag + ';I_'
						tag = tag + mwe.type

				elif active_mwe:
					tag = 'o_' + mwe.type
				else:
					tag = 'O'

				seqs_i.append((sents[s].tokens[t].text, sents[s].tokens[t].lemma,
                                sents[s].tokens[t].posTag, sents[s].tokens[t].dependencyParent,
                                sents[s].tokens[t].dependencyLabel, tag))
			seqs.append(seqs_i)
		return (seqs)
