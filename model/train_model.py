import pandas as pd
import numpy as np
import json
import keras
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def train_model():
	max_words = 500
	data = pd.read_csv("data.csv",sep='\t',skipinitialspace=True)
	train_x = [x[1] for x in data.values[:1000]]
	# index all the sentiment labels
	train_y = np.asarray([x[0] for x in  data.values[:1000]])

	tokenizer = Tokenizer(num_words=max_words)
	# feed tweets to the Tokenizer
	tokenizer.fit_on_texts(train_x)

	# Tokenizers come with a convenient list of words and IDs
	dictionary = tokenizer.word_index

	# Let's save this out so we can use it later
	with open('dictionary1.json', 'w') as dictionary_file:
		json.dump(dictionary, dictionary_file)

	allWordIndices = []
	# for each tweet, change each token to its ID in the Tokenizer's word_index
	for text in train_x:
		wordIndices = convert_text_to_index_array(text,dictionary)
		allWordIndices.append(wordIndices)

	# now we have a list of all tweets converted to index arrays.
	# cast as an array for future usage.
	allWordIndices = np.asarray(allWordIndices)

	# create one-hot matrices out of the indexed tweets
	train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
	# treat the labels as categories
	train_y = keras.utils.to_categorical(train_y, 2)

	model = Sequential()
	model.add(Dense(512, input_shape=(max_words,), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

	model.fit(train_x, train_y,
			batch_size=32,
			epochs=5,
			verbose=1,
			validation_split=0.1,
			shuffle=True)


	model_json = model.to_json()
	with open('model1.json', 'w') as json_file:
		json_file.write(model_json)

	model.save_weights('model1.h5')

def convert_text_to_index_array(text,dictionary):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

if __name__ == '__main__':
    train_model()








