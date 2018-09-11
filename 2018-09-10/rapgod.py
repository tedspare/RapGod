# rapgod.py
import numpy
from keras.layers import LSTM, Dense
from keras.models import Sequential

# the neural network architecture: one LSTM layer then one Dense layer
model = Sequential()
model.add(LSTM(32, input_shape=(1, 7), return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(Dense(1))

# build the model
model.compile(loss="binary_crossentropy",
			  optimizer="adam", 
			  metrics=["accuracy"])

# training data from part-of-speech sequences in rap lyrics
bars = numpy.genfromtxt("posIndex.csv", delimiter=",", dtype=int)

# break rap lines up into beginning and last word
first = bars[:, :7]
first = numpy.reshape(first, (first.shape[0], 1, first.shape[1]))

last = bars[:, -1:]
last = numpy.reshape(last, (last.shape[0], 1, last.shape[1]))

# train the model
model.fit(first, last,
		  batch_size=45, 
		  epochs=10, 
		  verbose=1,
		  validation_split=0.3)

