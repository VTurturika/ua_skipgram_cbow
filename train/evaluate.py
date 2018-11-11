import argparse
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('model', metavar='model', type=str, help='skipgram|cbow')
args = parser.parse_args()

if args.model != 'skipgram' and args.model != 'cbow':
    print('usage: python evaluate.py skipgram|cbow')
    sys.exit(0)

def read_dictionary():
    with open('models/{0}/metadata.tsv'.format(args.model), 'r') as file:
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


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/{0}/model.ckpt.meta'.format(args.model))
    saver.restore(sess, 'models/{0}/model.ckpt'.format(args.model))

    embeddings = tf.get_variable_scope().global_variables()[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm

    print('Write search queries (type q for quit):')
    query = input('query = ')
    while query != 'q':
        query = query.lower()
        if dictionary.get(query, -1) != -1:
            nearest = get_nearest(normalized_embeddings, word=query)
            nearest_words = [reversed_dictionary[id] for id in nearest]
            print('Nearest to {0}: {1}'.format(query, ', '.join(nearest_words)))
        else:
            print('unknown word')
        query = input('query = ')
