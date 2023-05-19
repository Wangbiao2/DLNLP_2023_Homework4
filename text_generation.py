import jieba
import numpy as np
import torch
import torch.nn as nn
import re
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras

DATA_PATH = '../data/'

def get_single_corpus(file_path):
    rr = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@★、…【】《》‘’[\\]^_`{|}~「」『』（）]+'  # 保留“”，。？！等字符
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(rr, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    words = list(jieba.cut(corpus))
    print("length: {}".format(len(words)))
    return words

def get_dataset(data):  # data为分词结果
    max_len = 50
    step = 3
    sentences = []
    next_tokens = []

    tokens = list(set(data))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    print('tokens:', len(tokens))
    for i in range(0, len(data) - max_len, step):
        sentences.append(
            list(map(lambda t: tokens_indices[t], data[i: i + max_len])))
        next_tokens.append(tokens_indices[data[i + max_len]])
    print('Number of sequences:', len(sentences))

    print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),))
        y[i] = 1
        next_tokens_one_hot.append(y)
    return sentences, next_tokens_one_hot, tokens, tokens_indices

callbacks_list = [
    keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
        filepath='model.h5',
        monitor='loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(  # 不再改善时降低学习率
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    keras.callbacks.EarlyStopping(  # 不再改善时中断训练
        monitor='loss',
        patience=3,
    ),
]

class SeqToSeq(nn.Module):
    def __init__(self, len_token, embedding_size):
        super(SeqToSeq, self).__init__()
        self.encode = nn.Embedding(len_token, embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size, 2, batch_first=True)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, len_token),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(x.shape)
        em = self.encode(x).unsqueeze(dim=1)
        print(em.shape)
        mid, _ = self.lstm(em)
        print(mid[:,0,:].shape)
        res = self.decode(mid[:, 0, :])
        print(res.shape)
        return res

def sample(preds, temperature=1.0):  # 预测的结果、温度
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train_and_test(x, y, tokens, tokens_indices, epochs=100):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = models.Sequential([
        layers.Embedding(len(tokens), 256),
        layers.LSTM(256),
        layers.Dense(len(tokens), activation='softmax')
    ])

    optimizer = optimizers.RMSprop(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)  # 交叉熵损失

    for e in range(epochs):

        model.fit(dataset, epochs=1, callbacks=callbacks_list)
        text = '他夫妻俩越争越大声。萧中慧再也忍耐不住，「啊」的一声，掩面奔出。萧中慧心中茫然一片，只觉眼前黑蒙蒙的，了无生趣。'
        print(text, end='')
        if e % 10 == 0:
            for temperature in [0.2, 0.5, 1.0]:
                text_cut = list(jieba.cut(text))[:60]
                print('\n temperature: ', temperature)
                print(''.join(text_cut), end='')
                for i in range(100):

                    sampled = np.zeros((1, 60))
                    for idx, token in enumerate(text_cut):
                        if token in tokens_indices:
                            sampled[0, idx] = tokens_indices[token]
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature=1)
                    next_token = tokens[next_index]
                    print(next_token, end='')

                    text_cut = text_cut[1: 60] + [next_token]

if __name__ == '__main__':
    flag = 0
    if flag == 0:
        file = DATA_PATH + '鸳鸯刀.txt'
        data = get_single_corpus(file)
        x, y, tokens, tokens_indices = get_dataset(data)
        train_and_test(x, y, tokens, tokens_indices,epochs=60)
    else:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        text = "他夫妻俩越争越大声。萧中慧再也忍耐不住，「啊」的一声，掩面奔出。萧中慧心中茫然一片，只觉眼前黑蒙蒙的，了无生趣。"
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        from transformers import pipeline, set_seed
        set_seed(55)
        generator = pipeline('text-generation', model='IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        generator("他夫妻俩越争越大声。萧中慧再也忍耐不住，「啊」的一声，掩面奔出。萧中慧心中茫然一片，只觉眼前黑蒙蒙的，了无生趣。", max_length=30, num_return_sequences=1)

