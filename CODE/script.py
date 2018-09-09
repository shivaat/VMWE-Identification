# coding: utf-8

import pickle
import numpy as np
import os
# fix random seed for reproducibility
np.random.seed(7)
from collections import Counter 
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, concatenate, Conv1D, BatchNormalization, CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from layers import ChainCRF

from gensim.models import *
from preprocess import *

import pickle, subprocess
from evaluation import labels2Parsemetsv


class Script(object):
	def __init__(self, lang, train, dev, test, word2vec_dir, model_name, initial_weight):
		self.lang = lang        # based on the language codes used in Parseme
		self.train = train      # name of the training file
		self.dev = dev          # name of the dev file. I dev is not going to be combined with train, the input for dev should be an empty string.
		self.test = test        # name of the test file
		self.word2vec_dir = word2vec_dir # the path to the pre-trained word2vec
		self.model_name = model_name     # name of the model 
		self.initial_weight = initial_weight # only when we want to continue training a model from a previous saved matrix of weights, otherwise we pass an empty string.

	def set_params(self, epoch=100, batch_size=100, pos=True):
		self.epoch = epoch
		self.batch_size = batch_size
		self.pos = pos 

	def encode(self, sents):
		"""integer encode the sentences"""
		t = Tokenizer(filters='\t\n', lower=False)
		t.fit_on_texts([" ".join(sent) for sent in sents])
		return t.word_index

	def read_train_test(self):
		### reading train and test ###
		self.train = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.train), 'rb'))
		self.X_train = [[x[0].lower() for x in elem] for elem in self.train]
		self.y_train = [[x[5] for x in elem] for elem in self.train]
		self.pos_train = [[x[2] for x in elem] for elem in self.train]
		if self.dev:
                        self.dev = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.dev), 'rb'))
                        self.X_train = self.X_train + [[x[0].lower() for x in elem] for elem in self.dev]
                        self.y_train = self.y_train + [[x[5] for x in elem] for elem in self.dev]
                        self.pos_train = self.pos_train + [[x[2] for x in elem] for elem in self.dev]
                        
                
		self.test = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.test), 'rb'))
		self.X_test = [[x[0].lower() for x in elem] for elem in self.test]
		self.y_test = [[x[5] for x in elem] for elem in self.test]
		self.pos_test = [[x[2] for x in elem] for elem in self.test]
		### ### ###

		self.words = list(set([elem for sublist in self.X_train+self.X_test for elem in sublist]))
		self.vocab_size = len(self.words) + 2 # because of <UNK> and <PAD> pseudo words
		self.n_classes = len(set([elem for sublist in (self.y_train+self.y_test) for elem in sublist])) + 1 # add 1 because of zero padding
		self.n_poses = len(set([elem for sublist in (self.pos_train+self.pos_test) for elem in sublist])) + 1
		print("number of POS: ",self.n_poses)

		self.max_length = len(max(self.X_train+self.X_test, key=len))
		print("max sentence length:", self.max_length)

		# assign a unique integer to each word/label
		self.w2idx = {word:i+1 for (i,word) in enumerate(self.words)}
		#w2idx = encode(X_train+X_test)
		self.l2idx = self.encode(self.y_train+self.y_test)
		self.pos2idx = self.encode(self.pos_train+self.pos_test)

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

		# keep the reverse to be able to decode back
		self.idx2w = {v: k for k, v in self.w2idx.items()}
		self.idx2l = {v: k for k, v in self.l2idx.items()}
		self.idx2pos = {v: k for k, v in self.pos2idx.items()}

		self.X_train_enc = [[self.w2idx[w] for w in sent] for sent in self.X_train]
		self.X_test_enc = [[self.w2idx[w] for w in sent] for sent in self.X_test]

		self.y_train_enc = [[self.l2idx[l] for l in labels] for labels in self.y_train]
		self.y_test_enc = [[self.l2idx[l] for l in labels] for labels in self.y_test]

		self.pos_train_enc = [[self.pos2idx[p] for p in poses] for poses in self.pos_train]
		self.pos_test_enc = [[self.pos2idx[p] for p in poses] for poses in self.pos_test]

		# zero-pad all the sequences 

		self.X_train_enc = pad_sequences(self.X_train_enc, maxlen=self.max_length, padding='post')
		self.X_test_enc = pad_sequences(self.X_test_enc, maxlen=self.max_length, padding='post') 

		self.y_train_enc = pad_sequences(self.y_train_enc, maxlen=self.max_length, padding='post')
		self.y_test_enc = pad_sequences(self.y_test_enc, maxlen=self.max_length, padding='post')

		self.pos_train_enc = pad_sequences(self.pos_train_enc, maxlen=self.max_length, padding='post')
		self.pos_test_enc = pad_sequences(self.pos_test_enc, maxlen=self.max_length, padding='post')

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

		if self.pos:
			self.pos_train_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_train_enc])
			self.pos_test_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_test_enc])
			print("pos array shape",self.pos_train_enc.shape)

		########## Access pre-trained embedding for the words list [START] ####
		self.wvmodel = KeyedVectors.load_word2vec_format("{}".format(self.word2vec_dir))

		self.embedding_dimension = self.wvmodel.vector_size + 7
		self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dimension))
		       
		UNKOWN = np.random.uniform(-1, 1, self.embedding_dimension)  # assumes that '<UNK>' does not exist in the embed vocab

		for word, i in self.w2idx.items():
		    if word in self.wvmodel.wv.vocab:
		        self.embedding_matrix[i] = np.concatenate((self.wvmodel.wv[word], wordShape(word))) 
		    else:
		        self.embedding_matrix[i] = UNKOWN
		        self.embedding_matrix[i][-7:] = wordShape(word)

		self.embedding_matrix[self.w2idx['<PAD>']] = np.zeros((self.embedding_dimension))

		self.embedding_layer = Embedding(self.embedding_matrix.shape[0],
		                            self.embedding_matrix.shape[1],
		                            weights=[self.embedding_matrix],
		                            trainable = False,
		                            name='embed_layer')
		####### Constructing our own embedding for the words list [END] #######

	def model_noPOS(self):  # ConvNet + LSTM
	    visible = Input(shape=(self.max_length,))
	    embed = self.embedding_layer(visible)
	    conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
	    conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
	    conc = concatenate([conv1, conv2])
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
	    output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)	
				# I tried without timeDistributed and I had a clear drop in results
	    model = Model(inputs=visible, outputs=output)
	    if self.initial_weight:
	        model.load_weights(self.initial_weight)
	    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
	    print(model.summary())
	    return model

	def model_withPOS(self):  # ConvNet + LSTM
	    visible = Input(shape=(self.max_length,))
	    embed = self.embedding_layer(visible)
	    # posInput = Input(shape=(max_length, 17))
	    posInput = Input(shape=(self.max_length,self.n_poses,))
	    embed = concatenate([embed, posInput])
	    conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
	    conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
	    conc = concatenate([conv1, conv2])
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
	    output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)  
	            # I tried without timeDistributed and I had a clear drop in results
	    model = Model(inputs=[visible, posInput], outputs=output)
	    if self.initial_weight:
	        model.load_weights(self.initial_weight)
	    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
	    print(model.summary())
	    return model

	def model_noPOS_CRF(self):  # ConvNet + LSTM +CRF
	    visible = Input(shape=(self.max_length,))
	    embed = self.embedding_layer(visible)
	    conv2 = Conv1D(200, 2, activation="relu", padding="same", name='conv2', 
	    					kernel_regularizer=regularizers.l2(0.001))(embed)
	    conv3 = Conv1D(200, 3, activation="relu", padding="same", name='conv3', 
	    					kernel_regularizer=regularizers.l2(0.001))(embed)
	    conc = concatenate([conv2, conv3])
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm'))(conc)
	    drop = Dropout(0.5)(lstm)
	    dense = TimeDistributed(Dense(self.n_classes, name='dense'))(drop) # This layer should not have any activation. 
	    crf = ChainCRF()
	    crf_output = crf(dense)
	    model = Model(inputs=visible, outputs=crf_output) 
	    if self.initial_weight:
	        model.load_weights(self.initial_weight)
	    model.compile(loss=crf.loss, optimizer='adam', metrics=['mae', 'acc'])
	    print(model.summary())
	    return model

	def model_withPOS_CRF(self):  ### ConvNet + LSTM + CRF with 2 inputs
		visible = Input(shape=(self.max_length,))
		embed = self.embedding_layer(visible)
		posInput = Input(shape=(self.max_length, self.n_poses,))
		embed = concatenate([embed, posInput])
		conv2 = Conv1D(200, 2, activation="relu", padding="same", name='conv2', 
							kernel_regularizer=regularizers.l2(0.001))(embed)
		conv3 = Conv1D(200, 3, activation="relu", padding="same", name='conv3', 
							kernel_regularizer=regularizers.l2(0.001))(embed)
		conc = concatenate([conv2, conv3])
		lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm'))(conc)
		drop = Dropout(0.5)(lstm)
		dense = TimeDistributed(Dense(self.n_classes, name='dense'))(drop) # This layer should not have any activation. 
		crf = ChainCRF()
		crf_output = crf(dense)
		model = Model(inputs=[visible, posInput], outputs=crf_output) 
		if self.initial_weight:
			model.load_weights(self.initial_weight)
		model.compile(loss=crf.loss, optimizer='adam', metrics=['mae', 'acc'])
		print(model.summary())
		return model

	def train_predict_test(self):
		self.model = getattr(self, self.model_name)()

		res_dir="./{}".format(self.lang)+"_"+self.model_name+"_results"
		if not os.path.exists(res_dir):
                        os.makedirs(res_dir)
		filepath = res_dir + "/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max', period=10)
		callbacks_list = [checkpoint]

		# since we are not using early stopping, we set validation split to 0
		#model.fit(X_train_enc, y_train_enc, validation_split=0, batch_size=100, epochs=50, callbacks=callbacks_list)
		if self.pos:
			self.model.fit([self.X_train_enc, 
						self.pos_train_enc], 
						self.y_train_enc, 
						validation_split=0, 
						batch_size=self.batch_size, 
						epochs=self.epoch, 
						callbacks=callbacks_list)
		else:
			self.model.fit(self.X_train_enc, 
						self.y_train_enc, 
						validation_split=0, 
						batch_size=self.batch_size, 
						epochs=self.epoch, 
						callbacks=callbacks_list)
		if self.pos:
			preds = self.model.predict([self.X_test_enc, self.pos_test_enc], batch_size=16, verbose=1)
		else:
			preds = self.model.predict(self.X_test_enc, batch_size=16, verbose=1)
		final_preds = []
		for i in range(len(self.X_test_enc)):
			pred = np.argmax(preds[i],-1)
			pred = [self.idx2l[p] for p in pred]
			final_preds.append(pred)
		predictionFileName = res_dir + '/predicted_{}'.format(self.lang)+'_'+self.model_name
		# save the predicted labels to a list
		with open(predictionFileName+'.pkl', 'wb') as f:
		    pickle.dump(final_preds, f)
		with open(predictionFileName+'.pkl', 'rb') as f:
		    labels1 = pickle.load(f)
		labels2Parsemetsv(labels1, '../{}/test.blind.cupt'.format(self.lang), predictionFileName+'_system.cupt')

		with open(res_dir +'/eval_'.format(self.lang)+self.model_name+'.txt', 'w') as f:
			f.write(subprocess.check_output(["../bin/evaluate.py", "--gold", "../{}/test.cupt".format(self.lang), "--pred", predictionFileName+"_system.cupt" ]).decode())
