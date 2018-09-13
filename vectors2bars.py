# vectors2bars.py

# takes an integer and returns a random word for the POS at that index
import json, random


def pos2random(index):

	# get vocab extracted from raps.txt
	with open("vocabPlus.json", "r") as f:
		vocab = json.load(f)
		posIndex = list(vocab.keys())

	# pick a random word from the options for that POS
	pos = posIndex[index - 1]
	pos = vocab[pos]

	options = list(pos.keys())
	word = random.choice(options)

	return word


if __name__ == "__main__":
	print(pos2random(16))
