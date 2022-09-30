from __future__ import print_function ###이것도 아래 언급한 소스코드에서 가져옴
### https://stackoverflow.com/questions/32032697/why-does-using-from-future-import-print-function-breaks-python2-style-print

import os
from time import process_time



# STAGE 1. Loading the Feature Representation Model: dna2vec
## Pretrained model: dna2vec
## https://github.com/pnpnpn/dna2vec

### 아래는 모델 소스코드 가져옴 + 구버전 코드 수정
### 코드 수정 레퍼런스
### https://stackoverflow.com/questions/42363897/attributeerror-type-object-word2vec-has-no-attribute-load-word2vec-format
### https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
### https://stackoverflow.com/questions/66868221/gensim-3-8-0-to-gensim-4-0-0
### https://radimrehurek.com/gensim/models/word2vec.html

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

d2v = MultiKModel(filepath)

t1_end = process_time()
print("Elapsed time of importing dna2vec in seconds:", t1_end - t1_start)

print("Embedding Shape:", d2v.vector('AAA').shape) ### (100, 0)




t2_start = process_time()

# STAGE 2. Defining Tokenizer & Embedding Functions

def tokenizer(sequence): ## 3-mer로 Tokenize
    token_list = []
    for i in range(len(sequence)-2): ## 마지막 3-mer까지 얻으려면 길이에서 2뺀 걸 마지막 starting index로 해야 함!
        token = sequence[i:i+3]
        token_list.append(token)

    return token_list


def tokenizeSeqs(dataset): ## 데이터셋을 Tokenize
    tokenset_list = []
    for sequence in dataset:
        token_list = tokenizer(sequence)
        tokenset_list.append(token_list)

    tokenset_array = np.array(tokenset_list) ### array화 - n131 참조
    return tokenset_array


def embedding(dataset): ## 데이터셋의 각 Token을 Embedding vectors로
    data_list = []
    for tokenset in dataset:
        embedding_list = []
        for token in tokenset:
            vector = d2v.vector(token)
            embedding_list.append(vector)
        data_list.append(embedding_list)

    dataset_array = np.array(data_list)
    return dataset_array




# STAGE 3. Data Loading, Tokenizing, & Embedding

import DataPreprocessing ## 따로 작성한 py 파일을 모듈로 import

X_train_pn, X_test_pn, y_train_pn, y_test_pn, X_train_sw, X_test_sw, y_train_sw, y_test_sw = DataPreprocessing.GetSets() 

X_train_tokenized_pn = tokenizeSeqs(X_train_pn)
X_train_embedded_pn = embedding(X_train_tokenized_pn)

X_test_tokenized_pn = tokenizeSeqs(X_test_pn)
X_test_embedded_pn = embedding(X_test_tokenized_pn)

X_train_tokenized_sw = tokenizeSeqs(X_train_sw)
X_train_embedded_sw = embedding(X_train_tokenized_sw)

X_test_tokenized_sw = tokenizeSeqs(X_test_sw)
X_test_embedded_sw = embedding(X_test_tokenized_sw)

### shape 확인
### --------------------
# print(X_train_pn.shape)
# print(X_train_tokenized_pn.shape)
# print(X_train_embedded_pn.shape)
# print(y_train_pn.shape)

# print(X_test_pn.shape)
# print(X_test_tokenized_pn.shape)
# print(X_test_embedded_pn.shape)
# print(y_test_pn.shape)

# print(X_train_sw.shape)
# print(X_train_tokenized_sw.shape)
# print(X_train_embedded_sw.shape)
# print(y_train_sw.shape)

# print(X_test_sw.shape)
# print(X_test_tokenized_sw.shape)
# print(X_test_embedded_sw.shape)
# print(y_test_sw.shape)
### --------------------

t2_end = process_time()
print("Elapsed time of loading and preprocessing data:", t2_end - t2_start)

 



# STAGE 4. Baseline Model (Chance Level)

### 훈련세트, 테스트세트 모두 클래스비율이 0.5로 동일하게 맞춰져 있음
### Chance Level = 0.5





# STAGE 5. Loading the Model Architecture of iEnhancer-CNN

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Activation

input = Input(shape=(198,100))
out = Conv1D(8, 7, strides=1, padding='same')(input)
out = Activation('relu')(out)
out = Conv1D(8, 7, strides=1, padding='same')(out)
out = Activation('relu')(out)
out = Dropout(0.5)(out)
out = Flatten()(out)
output = Dense(1, activation='sigmoid')(out)

model = Model(inputs=input, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=8*1e-4),
              metrics=['accuracy'])

### 참고) 크로스엔트로피 손실함수 'from_logits' 파라미터 관련
### https://stackoverflow.com/questions/61233425/what-should-i-use-as-target-vector-when-i-use-binarycrossentropyfrom-logits-tru

### 층별 shape 확인
# tf.keras.utils.plot_model(model, show_shapes=True)



### 학습해 봄
# t3_start = process_time()

# model.fit(X_train_embedded_pn, y_train_pn, batch_size=32, epochs=50, verbose=1)
# model.evaluate(X_test_embedded_pn, y_test_pn)

# t3_end = process_time()
# print("Elapsed time of loading and training the model:", t3_end - t3_start)



# STAGE 6. Fitting the Model by RandomizedSearchCV

## Early Stopping 지정
# checkpoint_filepath = "FMbest.hdf5"
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
# save_best = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
#     save_weights_only=True, mode='auto', save_freq='epoch', options=None)

# ## 논문에서 튜닝한 하이퍼파라미터 중 convolutional layers의 filter 수, 사이즈, 드롭아웃 비율을 cross-validation
# filter_number = [4, 8, 16, 32]
# filter_size = [3, 5, 7, 9]
# dropout_rate = [0.45, 0.5, 0.55]




