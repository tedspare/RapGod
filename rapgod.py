# rapgod.py

# trains an LSTM on 1x8 vectors corresponding to rap lines from raps.txt
import numpy, pronouncing
import matplotlib.pyplot as plot
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential, load_model
from vectors2bars import pos2random


# to build a better model, simply set train_mode=True and tweak hyperparameters
def ai(batch_size=5, train_mode=False):

	def train(batch_size):

		# the neural network architecture: one LSTM layer then two Dense layers
		model = Sequential()
		model.add(LSTM(128, input_shape=(5, 1), return_sequences=False))
		model.add(Dense(8))
		model.add(Dense(3))

		# build the model
		model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
		print(model.summary())

		# training data from part-of-speech sequences in rap lyrics
		bars = numpy.genfromtxt("posIndexPlus.csv", delimiter=",", dtype=int)

		# break rap lines up into beginning and last word
		first = bars[:, :5]
		first = first.reshape((first.shape[0], first.shape[1], 1))

		last = bars[:, -3:]
		last = last.reshape((last.shape[0], last.shape[1]))

		# train and save the model
		print("Training will take about {} seconds".format(first.shape[1] / 10))
		history = model.fit(first, last,
						  batch_size=batch_size,
						  epochs=20,
						  verbose=1,
						  validation_split=0.1)
		model.save("modelPlus.h5")

		return history


	def generate():

		# load the neural network and initialize it with a random sentence
		model = load_model("modelPlus.h5")
		start = numpy.random.randint(1, 36, size=(5))
		line = []

		# recursive function to generate n lines of rap
		def bar(model, seed, count=8, depth=0):

			if depth == count:
				return

			predict = model.predict(seed.reshape((1, 5, 1)))
			pred = predict.tolist()[0]
			start = numpy.append(seed[:2], predict)

			# wow more dense than the early universe. it means "add a word to the line for every output value"
			for i in pred:
				line.append(pos2random(int(round(i))).lower())

			return bar(model, start, count, depth + 1)


		bar(model, start)
		line = " ".join(line)
		print(line)
		return line

	if train_mode:
		history = train(batch_size)
		generate()
		return history
	else:
		generate()


if __name__ == "__main__":
	ai()
