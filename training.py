import time

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from load_data import prepare_data
from transformer import Transformer

clip = 1
lang1 = "eng"
lang2 = "fra"
SOS_token = 0
EOS_token = 1
PAD_token = 2
batch_size = 128
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(type, lang, sentence, max_length):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    if len(indexes) == max_length:
        return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)
    while len(indexes) < max_length:
        indexes.append(PAD_token)
    if type == "de":
        indexes = [SOS_token] + indexes
    return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)


def tensors_from_pair(input_lang, output_lang, pair, max_length_en, max_length_de):
    input_tensor = tensor_from_sentence("en", input_lang, pair[0], max_length_en)
    target_tensor = tensor_from_sentence("de", output_lang, pair[1], max_length_de)
    return input_tensor, target_tensor


input_lang, output_lang, pairs = prepare_data(lang1, lang2)
total_data = []
for i in range(0, len(pairs), batch_size):
    batch = []
    if i + batch_size < len(pairs):
        batch = pairs[i: i + batch_size]
    else:
        batch = pairs[i: len(pairs)]

    max_length_en = 0
    max_length_de = 0
    en_lengths = torch.zeros(batch_size,)
    for idx, pair in enumerate(batch):
        en_lengths[idx] = len(pair[0].split(" ")) + 1
        if len(pair[0].split(" ")) > max_length_en:
            max_length_en = len(pair[0].split(" "))
        if len(pair[1].split(" ")) > max_length_de:
            max_length_de = len(pair[1].split(" "))
    en_tensor, de_tensor = tensors_from_pair(input_lang, output_lang, batch[0], max_length_en + 1, max_length_de + 2)
    for idx, pair in enumerate(batch[1:]):
        en, de = tensors_from_pair(input_lang, output_lang, pair, max_length_en + 1, max_length_de + 2)
        en_tensor = torch.cat((en_tensor, en), 0)
        de_tensor = torch.cat((de_tensor, de), 0)
    total_data.append({
        "src": en_tensor.to(device),
        "trg": de_tensor.to(device),
        "src_len": en_lengths.to(device)
    })

total_data = total_data[:100]
train_data, test_data, y_train, y_test = train_test_split(total_data, np.zeros(len(total_data,)), test_size=0.2, random_state=42)

d_model = 512
heads = 8
N = 6
src_vocab = input_lang.n_words
trg_vocab = output_lang.n_words
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def create_masks(src, trg):
    input_seq = src
    input_pad = PAD_token
    input_msk = (input_seq != input_pad).unsqueeze(1)

    target_seq = trg
    target_pad = PAD_token
    target_msk = (target_seq != target_pad).unsqueeze(1)
    size = target_seq.size(1)  # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), 1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    target_msk = target_msk & nopeak_mask

    return input_msk, target_msk


def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):

        for i, batch in enumerate(train_data):
            src = batch["src"]
            trg = batch["trg"]
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next

            trg_input = trg[:, :-1]
            print("trg_input.shape ", trg_input.shape)

            # the words we are trying to predict

            targets = trg[:, 1:].contiguous().view(-1)

            # create function to make masks using mask code above

            src_mask, trg_mask = create_masks(src, trg_input)

            preds = model(src, trg_input, src_mask, trg_mask)

            print("preds.shape ", preds.view(-1, preds.size(-1)).shape)
            print("src_mask.shape ", targets.shape)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                   targets, ignore_index=PAD_token)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,% ds per %d iters" % ((time.time() - start) // 60,\
                epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()


train_model(10)
