from keras.layers import Embedding
from keras.layers import LSTM, Flatten, Activation, Input, GlobalMaxPool1D, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization, Convolution1D
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from model.callback import AnSelCB
from keras import regularizers

class LSTMModel:
    def __init__(self, dropout, use_bi, hidden_dim, pad, vocab_size, num_dimensions, emb_weights, trainable, use_pool, first_dense, lr):
        self.use_bi = use_bi
        self.hidden_dim = hidden_dim
        if self.use_bi:
            self.hidden_dim = hidden_dim / 2
        self.pad = pad
        self.vocab_size = vocab_size
        self.num_dimensions = num_dimensions
        self.emb_weights = emb_weights
        self.trainable = trainable
        self.use_pool = use_pool
        self.dropout = dropout
        self.first_dense = first_dense
        self.lr = lr
    
    def get_lstm_model(self):
        q1 = Input((self.pad,), dtype='int64', name='q1_base')
        q2 = Input((self.pad,), dtype='int64', name='q2_base')

        emb = Embedding(self.vocab_size, self.num_dimensions, weights=[self.emb_weights], trainable=self.trainable)
        if self.use_bi:
            model = Bidirectional(CuDNNLSTM(units=self.hidden_dim, return_sequences=self.use_pool))
        else:
            model = CuDNNLSTM(units=self.hidden_dim, return_sequences=self.use_pool)
        
        q1_emb = emb(q1)
        q1_emb = Dropout(self.dropout)(q1_emb)
        q1_emb = model(q1_emb)
        q1_emb = Dropout(self.dropout)(q1_emb)
        q1_emb = BatchNormalization()(q1_emb)

        q2_emb = emb(q2)
        q2_emb = Dropout(self.dropout)(q2_emb)
        q2_emb = model(q2_emb)
        q2_emb = Dropout(self.dropout)(q2_emb)
        q2_emb = BatchNormalization()(q2_emb)

        if self.use_pool:
            pool = GlobalMaxPool1D()
            q1_emb = pool(q1_emb)
            q2_emb = pool(q2_emb)
        
        merge = concatenate([q1_emb, q2_emb])
        merge = Dense(self.first_dense, activation='relu', kernel_regularizer=regularizers.l1(self.lr))(merge)
        merge = Dropout(self.dropout)(merge)
        merge = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l1(self.lr))(merge)
        trainining_model = Model(inputs=[q1, q2], outputs=merge, name='lstm_model')

        opt = Adam(self.lr)
        trainining_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return trainining_model