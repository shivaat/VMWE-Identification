# -*- coding: utf-8 -*-

import re, h5py
import numpy as np
from collections import Counter 
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

from corpus_reader import *

class Data(object):
	"""A preprocessor that prepares the data to be trained by the model 

	Attributes:
	    * lang_tr: name of the language (based on the language codes used in Parseme) for training data
	    * lang_dev: name of the language (based on the language codes used in Parseme) for development data
	    * lang_ts: name of the language (based on the language codes used in Parseme) for test data
	    * testORdev: whether we are getting results on test or development set
	    * word2vec_dir: path to the pre-trained word2vec
	    * elmo_dir: path to the pretrained elmo 
	    * model_name: name of the learning model 
	    
	"""
	def __init__(self, lang_tr, lang_dev, lang_ts, testORdev, word2vec_dir, elmo_dir, model_name, depAdjacency_gcn = 0, dep_info =False, pos=False):
		self.lang_tr = lang_tr
		self.lang_dev = lang_dev
		self.lang_ts = lang_ts 
		self.word2vec_dir = word2vec_dir
		self.elmo_dir = elmo_dir
		self.model_name = model_name
		self.testORdev = testORdev
		self.depInfo = dep_info
		self.depAdjacency_gcn = depAdjacency_gcn
		self.pos = pos

	def encode(self, sents):
		"""integer encode the sentences
		"""
		t = Tokenizer(filters='\t\n', lower=False)
		t.fit_on_texts([" ".join(sent) for sent in sents])
		return t.word_index

	def load_data(self, path):
		"""reading train and test 
		"""
		print("Reading the corpus .......")
		c = Corpus_reader(path+self.lang_tr+"/")
		train = c.read(c.train_sents)
		X_train = [[x[0].replace('.',"$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for elem in train]
		y_train = [[x[5] for x in elem] for elem in train]
		pos_train = [[x[2] for x in elem] for elem in train]
		self.dep_train = [[x[3] for x in elem] for elem in train]
		self.dep_tag_train = [[x[4] for x in elem] for elem in train]
		if self.testORdev == "TEST":    # self.lang != "EN" and   # Fr SHOMA and NAACL experiments train and test were combined here
			if self.lang_tr != self.lang_dev:
				c = Corpus_reader(path+self.lang_dev+"/")
				dev = c.read(c.train_sents)
			elif self.lang_dev != "EN":
				dev = c.read(c.dev_sents)
			else:
				dev = []
			#dev_file = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.dev), 'rb'))
			X_dev = [[x[0].replace('.',"$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for elem in dev]
			y_dev = [[x[5] for x in elem] for elem in dev]
			pos_dev = [[x[2] for x in elem] for elem in dev]
			self.dep_dev = [[x[3] for x in elem] for elem in dev] 
			self.dep_tag_dev = [[x[4] for x in elem] for elem in dev]    
		else:
			X_dev = []
			y_dev = []
			pos_dev = []
		
		if self.lang_ts != self.lang_dev:
			print("ERROR: Languages of DEV and TEST should be the same!")
			exit()			
		c = Corpus_reader(path+self.lang_ts+"/")
		if self.testORdev == "TEST":
			test = c.read(c.test_sents)
		elif self.testORdev == "DEV":
			test = c.read(c.dev_sents)
		elif self.testORdev == "CROSS_VAL":
			test = []
		else:
			print("ERROR: please specify if it is test or development!")  

		print("size of training: ", len(X_train))
		print("size of dev: ", len(X_dev))
		

		#test = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.test), 'rb'))
		X_test = [[x[0].replace('.',"$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for elem in test]
		y_test = [[x[5] for x in elem] for elem in test]
		pos_test = [[x[2] for x in elem] for elem in test]
		self.dep_test = [[x[3] for x in elem] for elem in test]
		self.dep_tag_test = [[x[4] for x in elem] for elem in test]
		### ### ###
		self.max_length = len(max(X_train+X_dev+X_test, key=len))
		
		print("size of test: ", len(X_test))
		print("max sentence length:", self.max_length)
		 
		######################################

		words = list(set([elem for sublist in X_train+X_dev+X_test for elem in sublist]))
		words = sorted(words)
		self.vocab_size = len(words) + 2 # because of <UNK> and <PAD> pseudo words
		self.n_classes = len(set([elem for sublist in (y_train+y_dev+y_test) for elem in sublist])) + 1 # add 1 because of zero padding
		self.n_poses = len(set([elem for sublist in (pos_train+pos_dev+pos_test) for elem in sublist])) + 1
		self.n_dep_tags = len(set([elem for sublist in self.dep_tag_train+self.dep_tag_dev+self.dep_tag_test for elem in sublist])) + 1
		print("number of POS: ",self.n_poses)
		print("number of dependency tags: ",self.n_dep_tags)

		# assign a unique integer to each word/label
		self.w2idx = {word:i+1 for (i,word) in enumerate(words)}
		#w2idx = encode(X_train+X_test)
		labels = list(set([elem for sublist in (y_train+y_dev+y_test) for elem in sublist]))
		labels = sorted(labels)
		self.l2idx = {l:i+1 for (i,l) in enumerate(labels)} 	#self.l2idx = self.encode(y_train+y_dev+y_test)
		poses = list(set([elem for sublist in (pos_train+pos_dev+pos_test) for elem in sublist]))
		poses = sorted(poses)
		self.pos2idx = {pos:i+1 for (i,pos) in enumerate(poses)}   #self.pos2idx = self.encode(pos_train+pos_dev+pos_test)
		deps = list(set([elem for sublist in (self.dep_tag_train + self.dep_tag_dev + self.dep_tag_test) for elem in sublist]))
		deps = sorted(deps)
		self.dep2idx = {dep:i+1 for (i,dep) in enumerate(deps)}
		#self.dep2idx = self.encode(self.dep_tag_train + self.dep_tag_dev + self.dep_tag_test)

		# encode() maps each word to a unique index, starting from 1. We additionally incerement all the 
		# values by 1, so that we can save space for 0 and 1 to be assigned to <PAD> and <UNK> later
		self.w2idx = Counter(self.w2idx)
		self.w2idx.update(self.w2idx.keys())
		self.w2idx = dict(self.w2idx) # convert back to regular dict (to avoid erroneously assigning 0 to unknown words)

		self.w2idx['<PAD>'] = 0
		self.w2idx['<UNK>'] = 1

		# on the label side we only have the <PADLABEL> to add
		self.l2idx['<PADLABEL>'] = 0
		self.pos2idx['<PADPOS>'] = 0
		self.dep2idx['<PADDEP>'] = 0

		# keep the reverse to be able to decode back
		self.idx2w = {v: k for k, v in self.w2idx.items()}
		self.idx2l = {v: k for k, v in self.l2idx.items()}
		self.idx2pos = {v: k for k, v in self.pos2idx.items()}
		self.idx2dep = {v: k for k, v in self.dep2idx.items()}

		self.X_train_enc = [[self.w2idx[w] for w in sent] for sent in X_train]
		self.X_dev_enc = [[self.w2idx[w] for w in sent] for sent in X_dev]
		self.X_test_enc = [[self.w2idx[w] for w in sent] for sent in X_test]

		self.y_train_enc = [[self.l2idx[l] for l in labels] for labels in y_train+y_dev]	# includes both train and dev
		self.y_test_enc = [[self.l2idx[l] for l in labels] for labels in y_test]


		self.pos_train_enc = [[self.pos2idx[p] for p in poses] for poses in pos_train+pos_dev]   # includes both train and dev
		self.pos_test_enc = [[self.pos2idx[p] for p in poses] for poses in pos_test]
		self.dep_train_enc = [[self.dep2idx[d] for d in deps] for deps in self.dep_tag_train+self.dep_tag_dev]
		self.dep_test_enc = [[self.dep2idx[d] for d in deps] for deps in self.dep_tag_test]

		# zero-pad all the sequences 

		self.X_train_enc = pad_sequences(self.X_train_enc, maxlen=self.max_length, padding='post')
		self.X_dev_enc = pad_sequences(self.X_dev_enc, maxlen=self.max_length, padding='post') 
		self.X_test_enc = pad_sequences(self.X_test_enc, maxlen=self.max_length, padding='post') 

		self.y_train_enc = pad_sequences(self.y_train_enc, maxlen=self.max_length, padding='post')
		self.y_test_enc = pad_sequences(self.y_test_enc, maxlen=self.max_length, padding='post')

		self.pos_train_enc = pad_sequences(self.pos_train_enc, maxlen=self.max_length, padding='post')
		self.pos_test_enc = pad_sequences(self.pos_test_enc, maxlen=self.max_length, padding='post')
		self.dep_train_enc = pad_sequences(self.dep_train_enc, maxlen=self.max_length, padding='post')
		self.dep_test_enc = pad_sequences(self.dep_test_enc, maxlen=self.max_length, padding='post')

		# one-hot encode the labels 
		self.idx = np.array(list(self.idx2l.keys()))
		self.vec = to_categorical(self.idx)
		self.one_hot = dict(zip(self.idx, self.vec))
		self.inv_one_hot = {tuple(v): k for k, v in self.one_hot.items()} # keep the inverse dict

		self.y_train_enc = np.array([[self.one_hot[l] for l in labels] for labels in self.y_train_enc])
		self.y_test_enc = np.array([[self.one_hot[l] for l in labels] for labels in self.y_test_enc])
		
		# one-hot encode the pos tags 
		self.idx = np.array(list(self.idx2pos.keys()))
		self.vec = to_categorical(self.idx)
		self.pos_one_hot = dict(zip(self.idx, self.vec))
		self.inv_pos_one_hot = {tuple(v): k for k, v in self.pos_one_hot.items()} # keep the inverse dict

		self.pos_train_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_train_enc])
		self.pos_test_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_test_enc])
		print("train pos array shape",self.pos_train_enc.shape) # pos information is not necessarily used by the model

		# one-hot encode the dependency tags 
		self.idx = np.array(list(self.idx2dep.keys()))
		self.vec = to_categorical(self.idx)
		self.dep_one_hot = dict(zip(self.idx, self.vec))
		self.inv_dep_one_hot = {tuple(v): k for k, v in self.dep_one_hot.items()} # keep the inverse dict

		self.dep_train_enc = np.array([[self.dep_one_hot[d] for d in deps] for deps in self.dep_train_enc])
		self.dep_test_enc = np.array([[self.dep_one_hot[d] for d in deps] for deps in self.dep_test_enc])
		print("train dep array shape",self.dep_train_enc.shape) # pos information is not necessarily used by the model



		#if self.depAdjacency_embed or self.depAdjacency_gcn:
		train_adjacency = self.load_adjacency(self.dep_train+self.dep_dev, 1) 
		#dev_adjacency = self.load_adjacency(self.dep_dev, 1)
		test_adjacency = self.load_adjacency(self.dep_test, 1)


		self.train_adjacency_matrices = [train_adjacency]
		#self.dev_adjacency_matrices = [dev_adjacency]
		self.test_adjacency_matrices = [test_adjacency]
		print("train adj shape", train_adjacency.shape)
		print("test adj shape", test_adjacency.shape)
		#self.train_adjacency_matrices = [np.concatenate((train_adjacency, dev_adjacency), axis=0)]
		#print("train adj shape", self.train_adjacency_matrices[0].shape)
		print("adjacency matrices size", len(self.train_adjacency_matrices))

		
		if self.elmo_dir:
			self.train_weights = self.load_elmo(X_train, self.lang_tr) 
			
			self.dev_weights = self.load_elmo(X_dev, self.lang_dev)
			self.train_weights = self.train_weights + self.dev_weights
			self.train_weights = np.array(self.train_weights, dtype = np.float32)  # we avoid concatenating np.arrays

			self.test_weights = self.load_elmo(X_test, self.lang_ts)    #, pos_test)
			self.test_weights = np.array(self.test_weights, dtype = np.float32)

			print("train weights shape: ", self.train_weights.shape)
			print("train weights type: ", self.train_weights.dtype)
			
			self.input_dim = len(self.train_weights[0][0])

		if self.word2vec_dir:
			self.load_word2vec()


	def load_word2vec(self):
		if not self.word2vec_dir:
			pass # do nothing if there is no path to a pre-trained embedding avialable  
		else:
			print("loading word2vec ...")
			wvmodel = KeyedVectors.load_word2vec_format("{}".format(self.word2vec_dir))

			embedding_dimension = wvmodel.vector_size 
			embedding_matrix = np.zeros((self.vocab_size, embedding_dimension))
			self.input_dim = embedding_dimension
			UNKOWN = np.random.uniform(-1, 1, embedding_dimension) 

			for word, i in self.w2idx.items():
			    if word in wvmodel.wv.vocab:
			        embedding_matrix[i] = wvmodel.wv[word] 
			    else:
			        embedding_matrix[i] = UNKOWN
			        #embedding_matrix[i][-7:] = self.word_shape(word)

			embedding_matrix[self.w2idx['<PAD>']] = np.zeros((embedding_dimension))

			self.embedding_layer = Embedding(embedding_matrix.shape[0],
			                            embedding_matrix.shape[1],
			                            weights=[embedding_matrix],
			                            trainable = False,
			                            name='embed_layer')

	def load_elmo(self, X, lang):	# the ultimate aim is to create a numpy array of shape (sent_num, max_sent_size, 1024)
									# here, we return a list and then after merging the list for train and dev we covert it to np.array
		if not self.elmo_dir:
			pass # do nothing if there is no path to a pre-trained elmo avialable 
		else:
			filename = self.elmo_dir + '/ELMo_{}'.format(lang)
			elmo_dict = h5py.File(filename, 'r')
			lst = []
			not_in_dict = 0
			for sent_toks in X:
				sent = "\t".join(sent_toks)
				if sent in elmo_dict:
				    item = list(elmo_dict[sent])	# ELMo representations for all words in the sentence
				else:
				    print("NO", sent, "is not in ELMO")
				    not_in_dict +=1
				    #item = []		
				    item = list(np.zeros((len(sent_toks), 1024)))
				min_lim = len(item)	#len(sent_toks)
				for i in range(min_lim, self.max_length):	# Here, we do padding, to make all sentences the same size
				    item.append([0]*1024)

				lst.append(item)
			if len(X):
				print('not found sentences:', not_in_dict)

			print('ELMO Loaded ...')
			return lst
			#return np.array(lst, dtype = np.float32)


	def load_headVectors(self, X, dep):

		filename = self.elmo_dir + '/ELMo_{}'.format(self.lang)
		elmo_dict = h5py.File(filename, 'r')
		lst = []
		sentIndx = 0
		for sent_toks, dep_toks in zip(X,dep):
			sent = "\t".join(sent_toks)
			if sent in elmo_dict:
				item = list(elmo_dict[sent])
			else:		
			    item = list(np.zeros((len(sent_toks), 1024)))
			min_lim = len(item)
			sent_deps = []
			for i in range(0,len(sent_toks)):
				if int(dep[sentIndx][i])>min_lim and min_lim!=0:
					print("error ", str(dep[sentIndx][i]), "greater than the sent length ", str(min_lim))
				if int(dep[sentIndx][i])-1:
					sent_deps.append(item[int(dep_toks[i]) - 1])
				else:	# the word is the root in the sent
					#item[i] = list(item[i]).extend([0]*1024)
					sent_deps.append(item[i])
			
			for i in range(min_lim, self.max_length):
				sent_deps.append([0]*1024)

			lst.append(sent_deps)
			sentIndx+=1
		print("dep head vectors shape: ", np.array(lst).shape)		
		return np.array(lst)

	def load_adjacency(self,dep, direction):

		if direction == 1:
			dep_adjacency = [self.adjacencyHead2Dep(d) for d in dep]
		elif direction == 0:
			dep_adjacency = [self.adjacencyDep2Head(d) for d in dep]
		elif direction == 3:
			dep_adjacency = [self.adjacencySelf(d) for d in dep]
		
		return np.array(dep_adjacency)

	def adjacencyDep2Head(self,sentDep):
		adjacencyMatrix = np.zeros((self.max_length,self.max_length), dtype=np.int)
		for i in range(len(sentDep)):
			if sentDep[i] != 0:
				adjacencyMatrix[i][sentDep[i]-1] = 1
				# adjacencyMatrix[sentDep[i]-1][i] = -1
		return adjacencyMatrix

	def adjacencyHead2Dep(self,sentDep):
		adjacencyMatrix = np.zeros((self.max_length,self.max_length), dtype=np.int)
		for i in range(len(sentDep)):
			if sentDep[i] != 0:
				#adjacencyMatrix[i][sentDep[i]-1] = 1
				adjacencyMatrix[sentDep[i]-1][i] = 1
		return adjacencyMatrix

	def adjacencySelf(self,sentDep):
		adjacencyMatrix = np.zeros((self.max_length,self.max_length), dtype=np.int)
		for i in range(len(sentDep)):
				adjacencyMatrix[i][i] = 1
		return adjacencyMatrix

	def get_pos_encoding(self, d_emb):
		"""outputs a position encoding matrix"""
		pos_enc = np.array([
			[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
			if pos != 0 else np.zeros(d_emb) 
				for pos in range(self.max_length)
				])
		pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
		pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
		return pos_enc
