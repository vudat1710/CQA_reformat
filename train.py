from utils.dataloader import PreprocessData
from utils.embedder import Embedder
from utils.vocabulary import Vocabulary
from utils.vectorizer import Vectorizer
from model.lstm import LSTMModel
from model.callback import AnSelCB, map_score

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class Trainer:
    def __init__(self, args):
        self.args = args

        self.q1s, self.q2s, self.labels = PreprocessData().build_corpus(self.args.trainpath)
        self.q1s_dev, self.q2s_dev, self.labels_dev = PreprocessData().build_corpus(self.args.devpath)
        self.vocab = Vocabulary(self.q1s, self.q2s).get_vocab()
        self.vect = Vectorizer(self.vocab, self.args.pad)
        self.emb_weights, self.num_dimensions = Embedder(self.vocab, self.args.pretrained_path, self.args.use_w2v, self.args.num_dimensions).embedder()
        self.lstm = LSTMModel(self.args.dropout, self.args.use_bi, self.args.hidden_dim, self.args.pad, len(self.vocab), self.num_dimensions, self.emb_weights, 
                    self.args.trainable, self.args.use_pool, self.args.first_dense_dim, self.args.lr).get_lstm_model()

    def helper(self):
        _l = []
        for filename in os.listdir(self.args.checkpoint_dir):
            if filename.endswith('.h5'):
                _l.append(int(filename[-5:-3]))
        if max(_l) >= 10:
            return str(max(_l))
        else:
            return str(' %d' % max(_l))

    def train(self):
        if not os.path.exists(self.args.checkpoint_dir):
            os.mkdir(self.args.checkpoint_dir)
        for filename in os.listdir(self.args.checkpoint_dir):
            os.remove(os.path.join(self.args.checkpoint_dir, filename))

        q1s_eb, q2s_eb = self.vect.vectorize(self.q1s, self.q2s)
        q1s_dev_eb, q2s_dev_eb = self.vect.vectorize(self.q1s_dev, self.q2s_dev)

        callback_list = [AnSelCB(self.q1s_dev, self.q2s_dev, self.labels_dev, [q1s_dev_eb, q2s_dev_eb], self.q1s, self.q2s, self.labels, [q1s_eb, q2s_eb]),
                            ModelCheckpoint('%s/model_improvement-{epoch:2d}.h5' % self.args.checkpoint_dir, monitor='map', verbose=1, save_best_only=True, mode='max'),
                            EarlyStopping(monitor='map', mode='max', patience=self.args.patience)]

        self.lstm.fit(
            [q1s_eb, q2s_eb],
            self.labels,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            validation_data=([q1s_dev_eb, q2s_dev_eb], self.labels_dev),
            verbose=1,
            callbacks=callback_list
        )

        lstm.summary()
    
    def test():
        print("Begin testing...")
        last_checkpoint = '{}/model_improvement-{}.h5'.format(self.args.checkpoint_dir, self.helper())
        self.lstm.load_weights(last_checkpoint)
        q1s_test, q2s_test, labels_test = PreprocessData().build_corpus(self.args.testpath)
        
        q1s_test_eb, q2s_test_eb = self.vect.vectorize(q1s_test, q2s_test)
        preds = self.lstm.predict([q1s_test_eb, q2s_test_eb])
        MAP, MRR = map_score(q1s_test, q2s_test, preds, labels_test)
        print("MAP: ", MAP)
        print("MRR: ", MRR)