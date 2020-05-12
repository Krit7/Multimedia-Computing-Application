# -*- coding: utf-8 -*-

import nltk
nltk.download('abc')
from nltk.corpus import abc, stopwords
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import string
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import itertools

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Reshape, dot

#load sentences and words
sents = abc.sents()

#make list
sentences = []
for s in sents:
    sentences.append(s)
    
del sents

#Words to remove from corpus
stop = stopwords.words('english')
punctuation = list(string.punctuation)

def check_punctuation(word):
    #checks if the word contains only punctuation marks
    if list(np.setdiff1d(np.array(list(word)), np.array(list(string.punctuation)))) == []:
        return True
    return False
    
def remove_extra(sentences, stop, punctuation):   
    i = 0
    words = []
    while i < len(sentences):
        print(i)
        extra = []
        for j in range(len(sentences[i])):
            if check_punctuation(sentences[i][j]) or sentences[i][j].lower() in stop:
                extra.append(sentences[i][j])
        temp = [w for w in sentences[i] if w not in extra]
        sentences[i] = temp
        #words.extend(sentences)
        i+=1
                
    for s in sentences:
        if len(s) <= 1:
            sentences.remove(s)
    words = list(set(list(itertools.chain.from_iterable(sentences))))
    return words, sentences

#Pre-process words and sentences
words, sentences = remove_extra(sentences, stop, punctuation)
#pickle.dump(words, open('cleaned_corpus.pkl', 'wb'))
#pickle.dump(sentences, open('cleaned_sentences.pkl', 'wb'))

def process(words, sentences):
    """Process raw inputs into a dataset."""
    count = Counter(words).most_common(len(words))
    
    dictionary = dict()
    for i in range(len(count)):
        word = count[i][0]
        dictionary[word] = i
    
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reversed_dictionary

#Index words
count, dictionary, reverse_dictionary = process(words, sentences)
sentences_enc = []
i = 0
while i < len(sentences): 
    enc = [dictionary[w] for w in sentences[i]]
    sentences_enc.append(enc)
    i+=1


def make_data(window, sentences, vocab_size):
    data_X = []
    data_Y = []
    for i in range(len(sentences)):
        print(i)
        for j in np.arange(0, len(sentences[i]), 3):
            start = max(j - window, 0)
            end = min(j + window + 1, len(sentences[i]))

            pos_window = np.array(list(set(range(start, end)) - set([i])))
            neg_window = np.array(list(set(range(vocab_size)) - set(sentences[i])))
            
            pos_word = sentences[i][random.sample(list(pos_window), 1)[0]]
            neg_word = random.sample(list(neg_window), 1)[0]
            
            data_X.append([sentences[i][j], pos_word])
            data_X.append([sentences[i][j], neg_word])
            data_Y.append(1)
            data_Y.append(0)
            
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
                
    return data_X, data_Y

window = 2
data_X, data_Y = make_data(window, sentences_enc, len(words))
target_data = data_X[:,0].reshape(len(data_X), 1)
context_data = data_X[:,1].reshape(len(data_X), 1)

dim = 300
vocab_len = len(words)

#Make Model
embedding = Embedding(vocab_len, dim, input_length=1, name='embedding')
input_target = Input((1,))
target = embedding(input_target)
target = Reshape((1, dim))(target)
input_context =Input((1,))
context = embedding(input_context)
context = Reshape((1, dim))(context)
dot_product = dot([target, context], axes = -1)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)
model = Model(input=[input_target, input_context], output=output)

emb_model = Model(input=input_target, output=target)

def tsne_plot_3d(title, label, embeddings, name, a=1):
    fig = plt.figure()
    plt.title(title)
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.savefig(name+'.png', format='png', dpi=150, bbox_inches='tight')
    #plt.show()

model.compile(loss='binary_crossentropy', optimizer='adam')
epochs = 40
    
for i in range(epochs):
    model.fit([target_data, context_data], data_Y, epochs = 1)
    
    embeddings = []
    for j in range(vocab_len):
        emb = emb_model.predict_on_batch(j)
        embeddings.append(emb[0][0])
        
    tsne_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=500, random_state=12)
    print('TSNE with 3 components')
    embeddings_3d = tsne_3d.fit_transform(embeddings[:5000])
    tsne_plot_3d('Visualizing Embeddings using t-SNE', 'NLTK ABC Corpus', embeddings_3d, 'plots/epoch_'+str(i)+'_3d', a=0.1)
