# rapgod.py

# trains an LSTM on 1x8 vectors corresponding to rap lines from raps.txt
import numpy
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential, load_model
import matplotlib.pyplot as plot
from vectors2bars import pos2random


def ai(batch_size=5, train_mode=False):

	def train(batch_size):

		# the neural network architecture: one LSTM layer then two Dense layers
		model = Sequential()
		model.add(LSTM(32, input_shape=(5, 1), return_sequences=False))
		model.add(Dense(8))
		model.add(Dense(3))

		# build the model
		model.compile(loss="mse",
					  optimizer="adam",
					  metrics=["accuracy"])

		# training data from part-of-speech sequences in rap lyrics
		bars = numpy.genfromtxt("posIndex.csv", delimiter=",", dtype=int)

		# break rap lines up into beginning and last word
		first = bars[:, :5]
		first = first.reshape((first.shape[0], first.shape[1], 1))

		last = bars[:, -3:]
		last = last.reshape((last.shape[0], last.shape[1]))

		# train the model
		history = model.fit(first, last,
						  batch_size=batch_size,
						  epochs=20,
						  verbose=1,
						  validation_split=0.1)

		# results
		scores = model.evaluate(first, last, verbose=1, batch_size=batch_size)
		print("Accuracy: {}".format(scores[1]))

		model.save("model.h5")

		return history


	def generate():

		model = load_model("model.h5")

		### temp
		bars = numpy.genfromtxt("posIndex.csv", delimiter=",", dtype=int)
		first = bars[:, :5]
		first = first.reshape((first.shape[0], first.shape[1], 1))

		x = first[0].reshape((1, 5, 1))
		predict = model.predict(x)

		# for line in range(10):
		pred = predict.tolist()[0]
		for i in pred:
			print(i)
			index = int(round(i))
			print(pos2random(index))


	if train_mode:
		history = train(batch_size)
		generate()
		return history
	else:
		generate()
		return


def graph():

	plot.title("Accuracy")
	plot.ylim((0, 1))
	plot.ion()
	plot.legend()

	for j in range(5):
		history = ai(2 * j + 10, train_mode=True)
		plot.plot(history.history["acc"], label=str(2 * j + 10))
		plot.ion()

	plot.show()


if __name__ == "__main__":
	# graph()
	for i in range(10):
		ai()
