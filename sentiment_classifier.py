# -*- coding: utf-8 -*-
"""Sentiment classification using keras

The main class of the module is SENTIMENTClassifier
 SENTIMENTClassifier.classify(request_list) parses JSON formatted requests and
returns a JSON formatted response containing the classifications.
"""
import json
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import keras.preprocessing.text as kpt

class SENTIMENTClassifier:
	def __init__(self, model_weights_path='model/model.h5', 
				dictionary_path='model/dictionary.json',
				model_path='model/model.json',num_words=500):
		self.tokenizer = Tokenizer(500)
		# for human-friendly printing
		self.labels = ['negative', 'positive']
		self.dictionary = self._load_dictionary(dictionary_path)
		self.loaded_model_json = self._load_model(model_path)
		self.model_weights = model_weights_path
		# and create a model from that
		self.model = model_from_json(self.loaded_model_json)
		# and weight your nodes with your saved values
		self.model.load_weights(self.model_weights)

	def _load_dictionary(self,dictionary_path):
		# read in our saved dictionary
		with open(dictionary_path, 'r') as dictionary_file:
			dictionary = json.load(dictionary_file)
		return dictionary

	def _load_model(self,model_path):
		# read in your saved model structure
		json_file = open(model_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		return loaded_model_json

		# this utility makes sure that all the words in your input
		# are registered in the dictionary
		# before trying to turn them into a matrix.
	def _convert_text_to_index_array(self,text,dictionary):
	    words = kpt.text_to_word_sequence(text)
	    wordIndices = []
	    for word in words:
	        if word in dictionary:
	            wordIndices.append(dictionary[word])
	        else:
	            print("'%s' not in training corpus; ignoring." %(word))
	    return wordIndices
	
	def classify(self,request_list):
		if(len(request_list) < 1):
			raise SENTIMENTClassifierInputError(
                'The minmum number of sentence in a single POST request is {}'
                .format(1))
		classifications = []
		for sentence in request_list:
			# format input for the neural net
			senArr = self._convert_text_to_index_array(sentence["sentence"],self.dictionary)
			input = self.tokenizer.sequences_to_matrix([senArr], mode='binary')
			# predict the sentiment
			pred = self.model.predict(input)
			# and print it
			print("%s sentiment; %f%% confidence" % (self.labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
			classifications.append({'sentence' : sentence["sentence"],
									'class' : self.labels[np.argmax(pred)], 
									'probability' : float(pred[0][np.argmax(pred)])
									})
		return classifications

class SENTIMENTClassifierError(Exception):
    """Base class for Exceptions raised by SENTIMENTClassifier"""


class SENTIMENTClassifierInputError(SENTIMENTClassifierError):
    """Exception raised for errors in the input."""



