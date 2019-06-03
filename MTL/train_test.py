import os, pickle, subprocess
import numpy as np
from keras.callbacks import ModelCheckpoint
from evaluation import labels2Parsemetsv

from sklearn.model_selection import KFold
from models.tag_models import Tagger 

from matplotlib import pyplot

class Train_Test():
	def __init__(self, pos, w2v, tagger_name, tagger, data, devORTest):
		self.pos = pos 
		self.w2v = w2v
		self.tagger_name = tagger_name
		self.tagger = tagger 
		self.data = data
		self.devORTest = devORTest

	def train(self, epoch, batch_size):
		self.res_dir="./results/"+self.tagger_name+"_results"
		
		if not os.path.exists(self.res_dir):
			os.makedirs(self.res_dir)
		filepath = self.res_dir + "/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max', period=10, save_weights_only=True)
		callbacks_list = [checkpoint]

		tr_inputs = []
		if "elmo" in self.tagger_name.lower():
			#inputs = [np.concatenate((self.data.train_weights, self.data.dev_weights), axis=0)]
			tr_inputs = [self.data.train_weights]   # which contains both train and dev
		else:  # if you need regular word2vec embeddings, pass X_train_enc and y_train_enc
			tr_inputs = [np.concatenate((self.data.X_train_enc, self.data.X_dev_enc), axis=0)]
		if self.w2v:
			tr_inputs += [np.concatenate((self.data.X_train_enc, self.data.X_dev_enc), axis=0)]
		if self.pos:
			tr_inputs += [self.data.pos_train_enc]
		if self.data.depAdjacency_gcn: # for this one only the concatenation has been done in the preprocessing
			tr_inputs += self.data.train_adjacency_matrices  
		tr_targets1 = self.data.y_train_enc   # np.concatenate((self.data.y_train_enc, self.data.y_dev_enc), axis = 0)
		tr_targets2 = self.data.train_adjacency_matrices[0]  # multi-tasking with dep adjacency significantly deteriorated the results for FA
		tr_targets3 = self.data.dep_train_enc
		if len(tr_inputs) == 1:
				self.history = self.tagger.fit(tr_inputs[0], 
				 				[tr_targets1, tr_targets2, tr_targets3], 
				 				validation_split=0.1, 
				 				batch_size=batch_size, 
				 				epochs=epoch)#, 
				 				#callbacks=callbacks_list)

		else:
				self.history = self.tagger.fit(tr_inputs, 
							   [tr_targets1, tr_targets2, tr_targets3], 
							   epochs=epoch,
							   validation_split=0.1, 
							   batch_size=batch_size)#, 
							   #callbacks=callbacks_list)
		self.tr_inputs = tr_inputs
		self.tr_targets1 = tr_targets1
		self.tr_targets2 = tr_targets2
		self.tr_targets3 = tr_targets3
		
		
	def test(self, data_path):
		ts_inputs = []
		if "elmo" in self.tagger_name.lower():
			ts_inputs = [self.data.test_weights]
		else:
			ts_inputs = [self.data.X_test_enc]
		if self.w2v:
			ts_inputs += [self.data.X_test_enc]
		if self.pos:
			ts_inputs += [self.data.pos_test_enc]
		if self.data.depAdjacency_gcn:
			ts_inputs += self.data.test_adjacency_matrices
		
		if len(ts_inputs)==1:
			preds, preds2, preds3 = self.tagger.predict(ts_inputs[0], batch_size=16, verbose=1)
		else:
			preds, preds2, preds3 = self.tagger.predict(ts_inputs, batch_size=16, verbose=1)
		
		self.ts_inputs = ts_inputs
		self.ts_targets1 = self.data.y_test_enc   # np.concatenate((self.data.y_train_enc, self.data.y_dev_enc), axis = 0)
		self.ts_targets2 = self.data.test_adjacency_matrices[0]  # multi-tasking with dep adjacency significantly deteriorated the results for FA
		self.ts_targets3 = self.data.dep_test_enc

		final_preds = []
		for i in range(len(self.data.X_test_enc)):
			pred = np.argmax(preds[i],-1)
			pred = [self.data.idx2l[p] for p in pred]
			final_preds.append(pred)
		prediction_file_name = self.res_dir + '/predicted_{}'.format(self.data.lang_ts)+'_'+self.tagger_name
		# save the predicted labels to a list
		with open(prediction_file_name+'.pkl', 'wb') as f:
		    pickle.dump(final_preds, f)
		
		with open(prediction_file_name+'.pkl', 'rb') as f:
		    labels1 = pickle.load(f)
		if self.devORTest == "TEST":	# we have DEV as part of training and are evaluating the test
			labels2Parsemetsv(labels1, data_path+'{}/test.blind.cupt'.format(self.data.lang_ts), prediction_file_name+'_system.cupt')

			with open(self.res_dir + '/eval'.format(self.data.lang_ts)+self.tagger_name+'.txt', 'w') as f:
				f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--train", data_path+"{}/train.cupt".format(self.data.lang_ts), "--gold", data_path+"{}/test.cupt".format(self.data.lang_ts), "--pred", prediction_file_name+"_system.cupt" ]).decode())
		else:
			labels2Parsemetsv(labels1, data_path+'/{}/dev.cupt'.format(self.data.lang_dev), prediction_file_name+'_system.cupt')

			with open(self.res_dir + '/eval'.format(self.data.lang_dev)+self.tagger_name+'.txt', 'w') as f:
				f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--train", data_path+"{}/train.cupt".format(self.data.lang_dev), "--gold", data_path+"{}/dev.cupt".format(self.data.lang_dev), "--pred", prediction_file_name+"_system.cupt" ]).decode())
	# summarize the performance of the fit model
	def summarize_model(self, model, history, train_inputs, train_targets, test_inputs, test_targets):
		# evaluate the model
		print(model.evaluate(train_inputs, train_targets, verbose=0))
		print(model.metrics_names)
		#_, train_acc1, train_acc2, train_acc3  = model.evaluate(train_inputs, train_targets, verbose=0)
		#_, test_acc = model.evaluate(test_inputs, test_targets, verbose=0)
		#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
		# plot loss during training
		pyplot.subplot(211)
		pyplot.title('Loss')
		pyplot.plot(history.history['loss'], label='train')
		#pyplot.plot(history.history['val_loss'], label='test')
		pyplot.legend()
		# plot accuracy during training
		pyplot.subplot(212)
		pyplot.title('Accuracy')
		pyplot.plot(history.history['output1_acc'], label='train')
		#pyplot.plot(history.history['val_acc'], label='test')
		pyplot.legend()
		pyplot.show()

	'''
	def cross_validation(self, epoch, batch_size, data_path):
		if self.devORTest == "CROSS_VAL": # or self.lang=='EN':  # when we want the results on DEV, we have DEV as the test data
			self.res_dir="./CROSSVAL_{}".format(self.lang)+"_"+self.tagger_name+"_results"
		else:
			pass
		if not os.path.exists(self.res_dir):
			os.makedirs(self.res_dir)

		kf = KFold(n_splits=5)
		i=0
		final_preds = [0]*len(self.data.X_train_enc)
		for train_index, test_index in kf.split(self.data.X_train_enc):
			print("Running Fold", i+1, "/", "5")
			X_train, X_test = self.data.X_train_enc[train_index], self.data.X_train_enc[test_index]
			pos_train, pos_test = self.data.pos_train_enc[train_index], self.data.pos_train_enc[test_index]
			y_train, y_test = self.data.y_train_enc[train_index], self.data.y_train_enc[test_index]
			#print(X_train.shape, y_train.shape, X_train[0])

			if "elmo" in self.tagger_name.lower():
				X_train, X_test = self.data.train_weights[train_index], self.data.train_weights[test_index]
				X_train_adj, X_test_adj = [], []
				if self.data.depAdjacency_gcn:
					for j in range(len(self.data.train_adjacency_matrices)):
						X_train_adj.append(self.data.train_adjacency_matrices[j][train_index])
						X_test_adj += [self.data.train_adjacency_matrices[j][test_index]]
				print(X_train.shape)


			model = None # Clearing the NN.
			#model = self.tagger
			model = Tagger(self.data, self.data.max_length, self.data.input_dim, self.data.n_poses, self.data.n_classes, "")
			model = getattr(model, self.tagger_name)() 
			#if "elmo" in self.tagger_name.lower():
			#	model.fit(train_text, y_train, validation_split=0, batch_size=10, epochs=1)

			if self.pos:
				model.fit([X_train, pos_train], 
							   y_train, 
							   epochs=epoch,
							   validation_split=0, 
							   batch_size=batch_size)
			elif self.data.depAdjacency_gcn:
				model.fit([X_train] + X_train_adj, 
							   y_train, 
							   epochs=epoch,
							   validation_split=0, 
							   batch_size=batch_size)
			else:
				model.fit(X_train, 
				 				y_train, 
				 				validation_split=0, 
				 				batch_size=batch_size, 
				 				epochs=epoch)
			i+=1

			#print('shape after reshape ', self.data.train_weights[0].reshape(1, -1).shape)
			#print('shape after reshape should be ', self.data.X_train_enc[0].reshape(1, -1).shape)
			print('shape that I have ', np.array([self.data.train_weights[0]]).shape)

			for t in test_index:
				if self.pos:
					pred = model.predict([np.array([self.data.train_weights[t]]), np.array([self.data.pos_train_enc[t]])])
				elif self.data.depAdjacency_gcn:
					pred = model.predict([np.array([self.data.train_weights[t]])] + [np.array([self.data.train_adjacency_matrices[j][t]]) for j in range(len(self.data.train_adjacency_matrices))])
				else:
					pred = model.predict(np.array([self.data.train_weights[t]])) #.reshape(1, -1))
				#print("prediction shape", pred.shape)
				pred = np.argmax(pred,-1)[0]
				#print("max prediction shape", pred.shape)
				pred = [self.data.idx2l[p] for p in pred]
				#print(pred)
				final_preds[t] = pred

		prediction_file_name = self.res_dir + '/predicted_{}'.format(self.lang)+'_'+self.tagger_name
		# save the predicted labels to a list
		with open(prediction_file_name+'.pkl', 'wb') as f:
		    pickle.dump(final_preds, f)
		with open(prediction_file_name+'.pkl', 'rb') as f:
		    labels1 = pickle.load(f)
		print("len(labels1)",len(labels1))
		labels2Parsemetsv(labels1, data_path+'{}/train.cupt'.format(self.lang), prediction_file_name+'_system.cupt')

		with open(self.res_dir + '/eval'.format(self.lang)+self.tagger_name+'.txt', 'w') as f:
			f.write(subprocess.check_output([data_path+"bin/evaluate_v1.py", "--train", data_path+"{}/train.cupt".format(self.lang), "--gold", data_path+"{}/train.cupt".format(self.lang), "--pred", prediction_file_name+"_system.cupt" ]).decode())
	'''
