import numpy as np
from gensim.models import Word2Vec
import math

class Embedder:
    def __init__(self, vocab, pretrained_path, use_w2v, num_dimensions=300):
        self.vocab = vocab
        self.num_dimensions = num_dimensions
        self.pretrained_path = pretrained_path
        self.use_w2v = use_w2v

    def get_w2v_details(self, filepath):
        model = Word2Vec.load(filepath)
        return (model.wv.vector_size, model.wv)

    def load_pretrained(self, filepath):
        print('Loading pretrained embedding data.........')
        pretrained = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                list_ = line.split(' ')
                word = list_[0]
                vect = np.array(list_[1:]).astype(np.float)
                self.num_dimensions = len(vect)
                pretrained[word] = vect
        print('Done loading pretrained')
        
        return pretrained

    def embedder(self):
        print('Creating embedding weights.....')
        if self.pretrained_path is not None:
            pretrained = self.load_pretrained(self.pretrained_path)
            if self.use_w2v:
                w2v_size, w2v_vects = self.get_w2v_details('skipgram_model.pkl')
                self.num_dimensions += w2v_size
                sd_rand = math.sqrt(3 / self.num_dimensions)
                embedding_weights = np.zeros((len(self.vocab), self.num_dimensions), dtype=float)
                for word in self.vocab:
                    if word in pretrained.keys():
                        if word in w2v_vects.vocab:
                            embedding_weights[int(self.vocab[word])] = np.concatenate((pretrained[word], w2v_vects[word]))
                        else:
                            embedding_weights[int(self.vocab[word])] = np.concatenate((pretrained[word], np.random.uniform(-sd_rand, sd_rand, w2v_size)))
                    else:
                        embedding_weights[int(self.vocab[word])] = np.random.uniform(-sd_rand, sd_rand, self.num_dimensions)
            else:
                sd_rand = math.sqrt(3 / self.num_dimensions)
                embedding_weights = np.zeros((len(self.vocab), self.num_dimensions), dtype=float)
                for word in self.vocab:
                    if word in pretrained.keys():
                        embedding_weights[int(self.vocab[word])] = pretrained[word]
                    else:
                        embedding_weights[int(self.vocab[word])] = np.random.uniform(-sd_rand, sd_rand, self.num_dimensions)
        else:
            sd_rand = math.sqrt(3 / self.num_dimensions)
            embedding_weights = np.zeros((len(self.vocab), self.num_dimensions), dtype=float)
            for word in self.vocab:
                embedding_weights[int(self.vocab[word])] = np.random.uniform(-sd_rand, sd_rand, self.num_dimensions)
        print('Done creating embedding weights')
        print ('Num dimensions is %d' % self.num_dimensions)
        return (embedding_weights, self.num_dimensions)


