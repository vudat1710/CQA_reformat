import numpy as np
from keras.preprocessing.sequence import pad_sequences

class Vectorizer:
    def __init__(self, vocab, pad):
        self.vocab = vocab
        self.pad = pad

    def sentence_to_vec(self, sentence):
        words = sentence.split(' ')
        vect = np.zeros(len(words), dtype=int)
        for i in range(len(words)):
            if words[i] in self.vocab:
                vect[i] = self.vocab[words[i]]
            else:
                vect[i] = self.vocab['_OOV_']
        
        return vect
    
    def vectorize(self, q1s, q2s):
        q1s = [self.sentence_to_vec(q1) for q1 in q1s]
        q2s = [self.sentence_to_vec(q2) for q2 in q2s]
        padded_q1s = pad_sequences(q1s, self.pad, padding='post', truncating='post')
        padded_q2s = pad_sequences(q2s, self.pad, padding='post', truncating='post')

        print('Sample vector: ', padded_q1s[0])

        return (padded_q1s, padded_q2s)

