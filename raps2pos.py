# raps2pos.py

# a collection of helper functions to prep rap lyrics
# first, from words to parts of speech then from parts of speech to vectors
# and a library of words for each part of speech
import nltk, csv, json
nltk.download("punkt")

posIndex = {'WDT': 32, 'NNS': 13, 'WP$': 34, 'VB': 26, 'VBD': 27, 'PRP': 18, 'JJR': 8, 'VBZ': 31, 'VBG': 28, 'NN': 12, 'DT': 3, 'WP': 33, 'UH': 25, 'VBP': 30, 'RB': 20, 'JJS': 9, 'RBR': 21, 'VBN': 29, 'LS': 10, 'CC': 1, 'TO': 24, 'RBS': 22, 'PDT': 16, 'MD': 11, 'PRP$': 19, 'WRB': 35, 'JJ': 7, 'EX': 4, 'IN': 6, 'POS': 17, 'NNPS': 15, 'CD': 2, 'FW': 5, 'RP': 23, 'NNP': 14}
posExamples = {'FW': {}, 'VBP': {}, 'CC': {}, 'WRB': {}, 'IN': {}, 'MD': {}, 'JJR': {}, 'LS': {}, 'NN': {}, 'NNP': {}, 'NNPS': {}, 'VB': {}, 'PDT': {}, 'VBG': {}, 'PRP': {}, 'RBR': {}, 'VBZ': {}, 'PRP$': {}, 'WDT': {}, 'TO': {}, 'CD': {}, 'JJ': {}, 'VBN': {}, 'RB': {}, 'POS': {}, 'VBD': {}, 'WP': {}, 'UH': {}, 'RP': {}, 'RBS': {}, 'WP$': {}, 'JJS': {}, 'NNS': {}, 'EX': {}, 'DT': {}}


def raps2pos(bars):

	with open("posPlus.csv", "w") as g:
		writer = csv.writer(g)

		for line in bars:
			tokens = nltk.word_tokenize(line)
			tags = nltk.pos_tag(tokens)
			pos = [tag[1] for tag in tags if "''" not in tag[1]]

			if len(pos) == 8:
				writer.writerow(pos)


def raps2index(bars):

	with open("posIndexPlus.csv", "w") as h:
		writer = csv.writer(h)

		for line in bars:
			tokens = nltk.word_tokenize(line)
			tags = nltk.pos_tag(tokens)
			pos = [tag[1] for tag in tags if "''" not in tag[1]]

			if "(" in pos:
				continue

			pos = [posIndex[p] for p in pos]
			if len(pos) == 8:
				writer.writerow(pos)


def vocab(bars):

	for line in bars:
		tokens = nltk.word_tokenize(line)
		tags = nltk.pos_tag(tokens)

		for tag in tags:
			if "''" not in tag[1]:
				pos = tag[1]

				if "(" in pos or ")" in pos:
					continue

				if tag[0] in posExamples[pos]:
					posExamples[pos][tag[0]] += 1
				else:
					posExamples[pos][tag[0]] = 1

	for key in posExamples.keys():
		total = sum(posExamples[key].values())

		for word in posExamples[key].keys():
			posExamples[key][word] /= total

	json.dump(posExamples, open("vocabPlus.json", "w"), sort_keys=True, indent=2)


def main(rap_file):

	with open(rap_file, "r") as f:
		bars = f.read()
		bars = bars.translate({ord(c): None for c in '!@#$().-,'})
		bars = bars.split("\n")

	print("Converting rap lines to part-of-speech tags")
	raps2pos(bars)

	print("Vectorizing lines of part-of-speech tags")
	raps2index(bars)

	print("Building vocabulary from rap lyrics")
	vocab(bars)


if __name__ == "__main__":
	main()
