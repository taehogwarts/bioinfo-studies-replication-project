from __future__ import print_function ###이것도 아래 언급한 소스코드에서 가져옴
### https://stackoverflow.com/questions/32032697/why-does-using-from-future-import-print-function-breaks-python2-style-print

import os
from time import process_time


# STAGE 1. Loading the Feature Representation Model: dna2vec
## Pretrained model: dna2vec
## https://github.com/pnpnpn/dna2vec

### 아래는 모델 소스코드 가져옴 + 구버전 코드 수정
### https://stackoverflow.com/questions/42363897/attributeerror-type-object-word2vec-has-no-attribute-load-word2vec-format
### https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
### https://stackoverflow.com/questions/66868221/gensim-3-8-0-to-gensim-4-0-0

t1_start = process_time()

### -----------------------------------------------------------------
import logbook
import tempfile
import numpy as np

# from gensim.models import word2vec
from gensim.models import KeyedVectors
from gensim import matutils

class SingleKModel:
    def __init__(self, model):
        self.model = model
        self.vocab_lst = sorted(model.key_to_index.keys())
        ###원 코드: self.vocab_lst = sorted(model.vocab.keys())

class MultiKModel:
    def __init__(self, filepath):
        self.aggregate = KeyedVectors.load_word2vec_format(filepath, binary=False)
        ###원 코드: self.aggregate = word2vec.Word2Vec.load_word2vec_format(filepath, binary=False)
        self.logger = logbook.Logger(self.__class__.__name__)

        vocab_lens = [len(vocab) for vocab in self.aggregate.key_to_index.keys()]
        ###원 코드: vocab_lens = [len(vocab) for vocab in self.aggregate.vocab.keys()]
        self.k_low = min(vocab_lens)
        self.k_high = max(vocab_lens)
        self.vec_dim = self.aggregate.vector_size

        self.data = {}
        for k in range(self.k_low, self.k_high + 1):
            self.data[k] = self.separate_out_model(k)

    def model(self, k_len):
        """
        Use vector('ACGTA') when possible
        """
        return self.data[k_len].model

    def vector(self, vocab):
        return self.data[len(vocab)].model[vocab]

    def unitvec(self, vec):
        return matutils.unitvec(vec)

    def cosine_distance(self, vocab1, vocab2):
        return np.dot(self.unitvec(self.vector(vocab1)), self.unitvec(self.vector(vocab2)))

    def l2_norm(self, vocab):
        return np.linalg.norm(self.vector(vocab))

    def separate_out_model(self, k_len):
        vocabs = [vocab for vocab in self.aggregate.key_to_index.keys() if len(vocab) == k_len]
        ###원 코드: vocabs = [vocab for vocab in self.aggregate.vocab.keys() if len(vocab) == k_len]
        if len(vocabs) != 4 ** k_len:
            self.logger.warn('Missing {}-mers: {} / {}'.format(k_len, len(vocabs), 4 ** k_len))

        header_str = '{} {}'.format(len(vocabs), self.vec_dim)
        with tempfile.NamedTemporaryFile(mode='w') as fptr:
            print(header_str, file=fptr)
            for vocab in vocabs:
                vec_str = ' '.join("%f" % val for val in self.aggregate[vocab])
                print('{} {}'.format(vocab, vec_str), file=fptr)
            fptr.flush()
            return SingleKModel(KeyedVectors.load_word2vec_format(fptr.name, binary=False))
            ###원 코드: return SingleKModel(word2vec.Word2Vec.load_word2vec_format(fptr.name, binary=False))
### -----------------------------------------------------------------

filepath = os.path.join(os.getcwd(), 'pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v')

model = MultiKModel(filepath)

t1_end = process_time()
print("Elapsed time of importing dna2vec in seconds:", t1_end - t1_start)



# STAGE 2. Loading the Achitecture of CNN
