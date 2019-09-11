class Vocabulary:
    def __init__(self, q1s, q2s):
        self.vocab = {}
        self.vocab['_PAD_'] = 0
        self.vocab['_OOV_'] = 1

        for q1 in q1s:
            q1_words = q1.split(' ')
            for word in q1_words:
                self.add_word(word)
        
        for q2 in q2s:
            q2_words = q2.split(' ')
            for word in q2_words:
                self.add_word(word)

        print('-------------------------------')
        print('Vocabulary with %d words created' % len(self.vocab))
        print('Sample words from vocab: ', self.vocab[2:50])
        print('-------------------------------')
        
    def add_word(self, word):
        if word not in self.vocab.keys():
            self.vocab[word] = len(self.vocab)
    
    def get_vocab(self):
        return self.vocab
    
