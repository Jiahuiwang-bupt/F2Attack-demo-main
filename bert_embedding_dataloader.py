import gzip
import os
import sys
import io
import re
import random
import csv
import numpy as np
import torch
csv.field_size_limit(sys.maxsize)
import logging
logger = logging.getLogger(__name__)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_corpus(path, csvf=False, clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    if not csvf:
        with open(path, encoding=encoding) as fin:
            for line in fin:
                if MR:

                    label, sep, text = line.partition(' ')
                    label = int(label)
                else:
                    label, sep, text = line.partition(',')
                    label = int(label) - 1
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text = line[0]
                label = int(line[1])
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels


def pad(sequences, pad_token='<pad>', pad_left=True):
    '''
        input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5, max(len(seq) for seq in sequences))
    if pad_left:
        return [[pad_token]*(max_len-len(seq)) + seq for seq in sequences]
    return [seq + [pad_token]*(max_len-len(seq)) for seq in sequences]


def attention_score_mask(sequences,  pad_left=True):
    '''
           input sequences is a list of text sequence [[str]]
           pad each text sequence to the length of the longest
    '''
    max_len = max(5, max(len(seq) for seq in sequences))
    attn_score_mask = []
    if pad_left:
        for seq in sequences:
            attention_score_mask = []
            for i in range(len(seq)):
                attention_score_mask.append(0)
            attn_padding = [float('-inf')] * (max_len - len(attention_score_mask))
            attention_score_mask += attn_padding
            attn_score_mask.append(attention_score_mask)
    return attn_score_mask


def attention_score_select(sequences,  pad_left=True):
    '''
           input sequences is a list of text sequence [[str]]
           pad each text sequence to the length of the longest
    '''
    max_len = max(5, max(len(seq) for seq in sequences))
    attn_score_select = []
    if pad_left:
        for seq in sequences:
            attention_score_mask = []
            for i in range(len(seq)):
                attention_score_mask.append(1)
            attn_padding = [0] * (max_len - len(attention_score_mask))
            attention_score_mask += attn_padding
            attn_score_select.append(attention_score_mask)
    return attn_score_select


def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    attn_mask_x = attention_score_mask(x)
    attn_select_x = attention_score_select(x)
    x = pad(x)

    length = len(x[0])
    batch_size = len(x)
    x = [map2id.get(w, oov_id) for seq in x for w in seq]
    x = torch.LongTensor(x)

    attn_mask_x = torch.FloatTensor(attn_mask_x)
    attn_select_x = torch.BoolTensor(attn_select_x)

    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda(), \
           attn_mask_x.cuda(), attn_select_x.cuda()


def create_one_batch_x(x, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda()


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [x[i] for i in lst]
    y = [y[i] for i in lst]

    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = []
    batches_y = []
    attn_score_mask_x = []
    attn_score_select_x = []
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by, attn_score_mask, attn_score_select = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        print("bx:", bx.size())
        print("by:", by.size())
        batches_x.append(bx)
        batches_y.append(by)
        attn_score_mask_x.append(attn_score_mask)
        attn_score_select_x.append(attn_score_select)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [batches_x[i] for i in perm]
        batches_y = [batches_y[i] for i in perm]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y, attn_score_mask_x, attn_score_select_x


class AttnInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_ids, attn_mask, attn_select):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.attn_mask = attn_mask
        self.attn_select = attn_select


def create_examples_features(x, y, map2id, max_seq_length):
    pad_token = '<pad>'
    oov = '<oov>'
    oov_id = map2id[oov]
    features = []
    for i in range(len(x)):
        text_ls = x[i]
        label = y[i]

        input_x_tokens = []
        input_x_ids = []
        attention_score_mask = []
        attention_score_select = []

        label = int(label)
        length = len(text_ls)

        for index in range(len(text_ls)):
            input_x_tokens.append(text_ls[index])
            attention_score_mask.append(0)
            attention_score_select.append(1)

        if len(text_ls) > max_seq_length:
            input_x_tokens = input_x_tokens[:max_seq_length]
            attention_score_mask = attention_score_mask[:max_seq_length]
            attention_score_select = attention_score_select[:max_seq_length]

        input_x_tokens_pad = [pad_token] * (max_seq_length-length)
        attention_score_mask_pad = [float('-inf')] * (max_seq_length-length)
        attention_score_select_pad = [0] * (max_seq_length-length)

        input_x_tokens += input_x_tokens_pad
        attention_score_mask += attention_score_mask_pad
        attention_score_select += attention_score_select_pad

        for j in range(len(input_x_tokens)):
            input_x_ids.append(map2id.get(input_x_tokens[j], oov_id))

        assert len(input_x_tokens) == max_seq_length
        assert len(input_x_ids) == max_seq_length
        assert len(attention_score_mask) == max_seq_length
        assert len(attention_score_select) == max_seq_length

        if i <= 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in input_x_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_x_ids]))
            logger.info("attention_score_mask: %s" % " ".join([str(x) for x in attention_score_mask]))
            logger.info("attention_score_select: %s" % " ".join([str(x) for x in attention_score_select]))
            logger.info("label: %s " % (label))

        features.append(
            AttnInputFeatures(input_ids=input_x_ids,
                              label_ids=label,
                              attn_mask=attention_score_mask,
                              attn_select=attention_score_select))
    return features


# shuffle training examples and create mini-batches
def create_batches_x(x, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx = create_one_batch_x(x[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]

    return batches_x


def load_embedding_npz(path):
    data = np.load(path)
    return [w.decode('utf8') for w in data['words']], data['vals']


def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = []
    vals = []
    with file_open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [float(x) for x in parts[1:]]
    return words, np.asarray(vals).reshape(len(words), -1)


def load_bert_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = []
    vals = []
    with file_open(path, encoding='utf-8') as fin:
        lines = fin.readlines()
        lines = lines[1:]
        for line in lines:
            word, sep, embedding = line.partition(' ')
            embedding = embedding.strip('\n').strip(' ')
            # print(embedding)
            embedding = embedding.split(' ')
            words.append(word)
            vals += [float(x) for x in embedding]
    return words, np.asarray(vals).reshape(len(words), -1)


def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    elif path.endswith(".txt"):
        return load_bert_embedding_txt(path)
    else:
        return load_embedding_txt(path)
