# system
import random
import numpy as np
import chainer

# saves model at each trigger, by default at each epoch
# it has by default very low priority, which means that model will not be saved
# if the training is stopped by another extension
@chainer.training.make_extension(trigger=(1, 'epoch'), priority=-100)
def model_saver(trainer):
    model = trainer.updater.get_optimizer("main").target
    chainer.serializers.save_npz("{}/model_tmp".format(trainer.out), model)

# iterator which generates data from randomly sorted sequences.
# data from each sequence are in the correct order, only the sequences as a whole are shuffled
class SequenceShuffleIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, sentence_offsets, batch_size, repeat=True, shuffle=True):
        if sentence_offsets is not None and len(dataset) != sentence_offsets[-1]:
            raise ValueError("Last sentence offset must be equal to the dataset length")
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.shuffle = shuffle
        self.offsets = [i * len(dataset) // batch_size for i in range(batch_size)]
        self.sentence_offsets = sentence_offsets
        self.iteration = 0
        if self.shuffle:
            self.reshuffle()
    
    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_data = self.get_data()
        self.iteration += 1
        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            if self.shuffle:
                self.reshuffle()
        return cur_data

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def serialize(self, serializer):
        self.iteration = serializer("iteration", self.iteration)
        self.epoch = serializer("epoch", self.epoch)

    def get_data(self):
        if self.shuffle:
            return [self.dataset[self.shuffled_idx[(offset + self.iteration) % len(self.dataset)]] for offset in self.offsets]
        else:
            return [self.dataset[(offset + self.iteration) % len(self.dataset)] for offset in self.offsets]
    
    def reshuffle(self):
        self.shuffled_idx = np.zeros(self.sentence_offsets[-1], dtype=np.int32)
        shuffled_sentences = list(range(len(self.sentence_offsets) - 1))
        random.shuffle(shuffled_sentences)
        l_offset = 0
        for s in shuffled_sentences:
            l = self.sentence_offsets[s+1] - self.sentence_offsets[s]
            idx = np.arange(l, dtype=np.int32) + self.sentence_offsets[s]
            self.shuffled_idx[l_offset:l_offset+l] = idx
            l_offset = l_offset + l

# back-propagation through time updater
class BPTTUpdater(chainer.training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, **kwarg):
        super(BPTTUpdater, self).__init__(train_iter, optimizer, **kwarg)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        for _ in range(self.bprop_len):
            batch = train_iter.__next__()
            x, t = self.converter(batch)
            x = chainer.cuda.to_gpu(x)
            t = chainer.cuda.to_gpu(t)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()