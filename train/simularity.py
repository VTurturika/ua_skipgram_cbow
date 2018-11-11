import argparse
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('model', metavar='model', type=str, help='skipgram|cbow')
parser.add_argument('--version', metavar='version', type=str, help='mm.dd-hh:mm:ss')
args = parser.parse_args()

if args.model != 'skipgram' and args.model != 'cbow':
	print('usage: python evaluate.py skipgram|cbow')
	sys.exit(0)

if args.version:
	model_path = 'models/{0}.{1}'.format(args.model, args.version)
else:
	model_path = 'models/{0}'.format(args.model)


def read_tests():
	with open('tests/simularity.txt', 'r') as file:
		return [line.split() for line in file.readlines()]


tests = read_tests()


def read_dictionary():
	with open('{0}/metadata.tsv'.format(model_path), 'r') as file:
		words = file.read().split()
		dictionary = {}
		for (i, word) in enumerate(words):
			dictionary[word] = i
		reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reversed_dictionary


dictionary, reversed_dictionary = read_dictionary()


def get_nearest(embeddings, word=None, embedding=None):
	if word != None:
		word_embedding = tf.nn.embedding_lookup(embeddings, [dictionary.get(word, 0)])
	else:
		word_embedding = embedding
	similarity = tf.matmul(word_embedding, embeddings, transpose_b=True)
	sim = similarity.eval()
	nearest = (-sim).argsort()[0]
	return nearest[1:11]


def calculate_score(actual, expected):
	score = 0
	for expected_word in expected:
		if expected_word in actual:
			score += 0.25
	return score


with tf.Session() as sess:
	saver = tf.train.import_meta_graph('{0}/model.ckpt.meta'.format(model_path))
	saver.restore(sess, '{0}/model.ckpt'.format(model_path))

	embeddings = tf.get_variable_scope().global_variables()[0]
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
	normalized_embeddings = embeddings / norm

	total = 0
	for test in tests:
		if dictionary.get(test[0], -1) != -1:
			nearest = get_nearest(normalized_embeddings, word=test[0])
			nearest_words = [reversed_dictionary[id] for id in nearest]
			score = calculate_score(nearest_words, test[1:])
			total += score
			print('Word {0}:'.format(test[0]))
			print('Expected: ' + ', '.join(test[1:]))
			print('Actual: ' + ', '.join(nearest_words))
			print('Score = ' + str(score))
	print('Total score = ' + str(total))