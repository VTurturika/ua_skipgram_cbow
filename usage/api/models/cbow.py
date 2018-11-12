import argparse
import sys
import tensorflow as tf


def read_dictionary():
    with open('models/cbow/metadata.tsv', 'r') as file:
        words = file.read().split()
        dictionary = {}
        for (i, word) in enumerate(words):
            dictionary[word] = i
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reversed_dictionary

def evaluate(embeddings, word=None, embedding=None):
    if word != None:
        word_embedding = tf.nn.embedding_lookup(embeddings, [dictionary.get(word, 0)])
    else:
        word_embedding = embedding
    similarity = tf.matmul(word_embedding, embeddings, transpose_b=True)
    sim = similarity.eval()
    nearest = (-sim).argsort()[0]
    return nearest[1:11]


dictionary, reversed_dictionary = read_dictionary()

def get_nearest_cbow(word):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/cbow/model.ckpt.meta')
        saver.restore(sess, 'models/cbow/model.ckpt')

        embeddings = tf.get_variable_scope().global_variables()[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        if dictionary.get(word, -1) == -1:
            return 'unknown word'
        nearest = evaluate(normalized_embeddings, word=word)
        nearest_words = [reversed_dictionary[id] for id in nearest]
        return nearest_words
