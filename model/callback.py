from keras.callbacks import Callback

def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
    QA_pairs = {}
    for i in range(len(s1s_dev)):
        pred = y_pred[i]
        s1 = s1s_dev[i]
        s2 = s2s_dev[i]
        if s1 in QA_pairs:
            QA_pairs[s1].append((s2, labels_dev[i], pred))
        else:
            QA_pairs[s1] = [(s2, labels_dev[i], pred)]

    MRR = 0
    MAP = []
    for s1 in QA_pairs.keys():
        p = 0
        AP = []
        MRR_check = False

        QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)
        for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True
                p += 1
                AP.append(p / (idx + 1))
        if(len(AP) > 0):
            MAP.append(sum(AP) / len(AP))
    MRR /= len(MAP)
    return sum(MAP) / len(MAP), MRR

class AnSelCB(Callback):
    def __init__(self, val_q, val_s, y, inputs, train_q, train_s, train_y, train_inputs):
        self.val_q = val_q
        self.val_s = val_s
        self.val_y = y
        self.val_inputs = inputs
        self.train_q = train_q
        self.train_s = train_s
        self.train_y = train_y
        self.train_inputs = train_inputs
    
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        train_pred = self.model.predict(self.train_inputs)
        map__, mrr__ = map_score(self.val_q, self.val_s, pred, self.val_y)
        train_map, train_mrr = map_score(self.train_q, self.train_s, train_pred, self.train_y)
        print('val MRR %f; val MAP %f' % (mrr__, map__))
        print('train MRR %f; train MAP %f' % (train_mrr, train_map))
        logs['mrr'] = mrr__
        logs['map'] = map__