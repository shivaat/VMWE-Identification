import pickle, os
import numpy as np
from collections import Counter 
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, \
						concatenate, Conv1D, BatchNormalization, CuDNNLSTM, Lambda, \
						Multiply, Add, Activation, Flatten, MaxPooling1D 
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.regularizers import l2
import keras.initializers
from models.layers import * 

class Tagger(object):
	def __init__(self, data, initial_weight=''):
		self.max_length = data.max_length
		self.n_poses = data.n_poses
		self.n_classes = data.n_classes
		self.initial_weight = initial_weight
		self.data = data

	def model_ELMo(self):
		#visible = Input(shape=(self.max_length,), dtype="string") #tf.string
		#embed = Lambda(self.ElmoEmbedding, output_shape=(None,1024,))(visible)
		embed = Input(shape=(self.max_length,self.data.input_dim))
		# conv1 = Conv1D(200, 1, activation="relu", padding="same", name='conv1')(embed)   # adding this made the results worse
		conv2 = Conv1D(200, 2, activation="relu", padding="same", name='conv2')(embed)
		conv3 = Conv1D(200, 3, activation="relu", padding="same", name='conv3')(embed)
		conc = BatchNormalization()(concatenate([conv2, conv3]))
		
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		else:
			lstm = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm'))(conc)
			lstm = Dropout(0.5)(lstm)

		output_depTag = TimeDistributed(Dense(self.data.n_dep_tags, activation='softmax', name='dense2'), name='time_dist2')(lstm)
		output_dep = TimeDistributed(Dense(self.max_length, activation='softmax', name='dense3'), name='time_dist3')(lstm)
		lstm = Bidirectional(LSTM(100,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(lstm)
		output1 = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense1'), name='output1')(lstm)   # I tried output_dep as the input 
		                                                                                                           	   # to this layer, and it was awful	
		#output = Dense(n_classes, activation='softmax', name='dense')(embed)	
		model = Model(inputs=embed, outputs=[output1, output_dep, output_depTag])
		## The following was used to get the results for the submission
		#model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[2., 0.5, 0.5], metrics=['mae', 'acc'])
		model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['mae', 'acc']) 
		print(model.summary())
		return model	

	def model_ELMo_LSTM_withPOS(self):
		#visible = Input(shape=(self.max_length,), dtype="string") #tf.string
		#embed = Lambda(self.ElmoEmbedding, output_shape=(None,1024,))(visible)
		elmo_embed = Input(shape=(self.max_length,self.data.input_dim))
		posInput = Input(shape=(self.max_length,self.n_poses,))
		embed = concatenate([elmo_embed, posInput])
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm1', dropout=0.5, recurrent_dropout=0.2))(embed)
			lstm = Bidirectional(LSTM(200,return_sequences=True, name='lstm2', dropout=0.5, recurrent_dropout=0.2))(lstm)
			output_depTag = TimeDistributed(Dense(self.data.n_dep_tags, activation='softmax', name='dense2'), name='time_dist2')(lstm)
			output_dep = TimeDistributed(Dense(self.max_length, activation='softmax', name='dense3'), name='time_dist3')(lstm)
			lstm = Bidirectional(LSTM(100,return_sequences=True, name='lstm3', dropout=0.5, recurrent_dropout=0.2))(lstm)
		else:
			lstm = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm1'))(embed)
			lstm = Dropout(0.5)(lstm)
			lstm = Bidirectional(CuDNNLSTM(200,return_sequences=True, name='lstm2'))(lstm)
			lstm = Dropout(0.5)(lstm)
			output_depTag = TimeDistributed(Dense(self.data.n_dep_tags, activation='softmax', name='dense2'), name='time_dist2')(lstm)
			output_dep = TimeDistributed(Dense(self.max_length, activation='softmax', name='dense3'), name='time_dist3')(lstm)
			lstm = Bidirectional(CuDNNLSTM(100,return_sequences=True, name='lstm3'))(lstm)
			lstm = Dropout(0.5)(lstm)
			
		output1 = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='output1')(lstm)	
		#output = Dense(n_classes, activation='softmax', name='dense')(embed)	
		model = Model(inputs=[elmo_embed, posInput], outputs=[output1, output_dep, output_depTag])
		model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[2., 0.5, 0.5], metrics=['mae', 'acc'])
		print(model.summary())
		return model

	def model_withPOS_closed(self):
	    visible = Input(shape=(self.max_length,))
	    embed = Embedding(output_dim=512, input_dim=self.data.vocab_size, input_length=self.max_length)(visible)
	    # posInput = Input(shape=(max_length, 17))
	    posInput = Input(shape=(self.max_length,self.n_poses,))
	    embed = BatchNormalization()(concatenate([embed, posInput]))
	    conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
	    conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
	    conc = BatchNormalization()(concatenate([conv1, conv2]))
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
	    #lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(lstm) # Having an LSTM here deteriorated the results significantly
	    output_depTag = TimeDistributed(Dense(self.data.n_dep_tags, activation='softmax', name='dense2'), name='time_dist2')(lstm)
	    output_dep = TimeDistributed(Dense(self.max_length, activation='softmax', name='dense3'), name='time_dist3')(lstm)
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(lstm) 
	    lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(lstm)
	    output1 = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='output1')(lstm)  
	            # I tried without timeDistributed and I had a clear drop in results
	    model = Model(inputs=[visible, posInput], outputs=[output1, output_dep, output_depTag])
	    model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[2., 0.5, 0.5], metrics=['mae', 'acc'])
	    print(model.summary())
	    return model

	def model_ELMo_withPOS_plus(self):
		#visible = Input(shape=(self.max_length,), dtype="string") #tf.string
		#embed = Lambda(self.ElmoEmbedding, output_shape=(None,1024,))(visible)
		elmo_embed = Input(shape=(self.max_length,self.data.input_dim))
		posInput = Input(shape=(self.max_length,self.n_poses,))
		embed = concatenate([elmo_embed, posInput])
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		if self.max_length < 300:
			lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		else:
			lstm = Bidirectional(CuDNNLSTM(300,return_sequences=True, name='lstm'))(conc)
			lstm = Dropout(0.5)(lstm)

		output_depTag = TimeDistributed(Dense(self.data.n_dep_tags, activation='softmax', name='dense2'), name='time_dist2')(lstm)
		output_dep = TimeDistributed(Dense(self.max_length, activation='softmax', name='dense3'), name='time_dist3')(lstm)
		lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm2', dropout=0.5, recurrent_dropout=0.2))(lstm)
		output1 = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='output1')(lstm)	
		#output = Dense(n_classes, activation='softmax', name='dense')(embed)	
		model = Model(inputs=[elmo_embed, posInput], outputs=[output1, output_dep, output_depTag])
		## The following was used to get the results for the submission
		#model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[2., 0.5, 0.5], metrics=['mae', 'acc']) 
		model.compile(optimizer='Adam', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['mae', 'acc']) 
		print(model.summary())
		return model

	def model_ELMo_withW2V(self):
		elmo_embed = Input(shape=(self.max_length,self.data.input_dim))
		visible = Input(shape=(self.max_length,))
		embed = self.data.embedding_layer(visible)
		embed = concatenate([elmo_embed, embed])
		conv1 = Conv1D(200, 2, activation="relu", padding="same", name='conv1')(embed)
		conv2 = Conv1D(200, 3, activation="relu", padding="same", name='conv2')(embed)
		conc = concatenate([conv1, conv2])
		lstm = Bidirectional(LSTM(300,return_sequences=True, name='lstm', dropout=0.5, recurrent_dropout=0.2))(conc)
		output = TimeDistributed(Dense(self.n_classes, activation='softmax', name='dense'), name='time_dist')(lstm)  
		model = Model(inputs=[elmo_embed, visible], outputs=output)
		if self.initial_weight:
		    model.load_weights(self.initial_weight)
		model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mae', 'acc'])
		print(model.summary())
		return model



		################################################
		######### NON-ELMo models from before ##########
		################################################

	def model_withPOS(self):  # ConvNet + LSTM (SHOMA: the shared task model)
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

