import os
import torch
import bert_embedding_dataloader
import dataloader
import argparse
import torch.nn as nn
import modules
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
import sys
import numpy as np

no_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu")


class AttnInputFeatures(object):

    def __init__(self, input_ids, attn_mask, attn_select):
        self.input_ids = input_ids
        # self.label_ids = label_ids
        self.attn_mask = attn_mask
        self.attn_select = attn_select


class Attn_Model(nn.Module):
    def __init__(self, model_dir, bert_embedding, glove_embedding, max_seq_length, hidden_size, depth, dropout, nclasses):
        super(Attn_Model, self).__init__()
        self.attn_model = Joint_Embedding_Model(bert_embedding, glove_embedding, hidden_size, depth, dropout, nclasses)
        self.attn_model.load_state_dict(torch.load(model_dir, map_location='cuda:0'))
        self.dataset = Attn_Dataset(self.attn_model.glove_word2id, max_seq_length, batch_size=1)

    def text_pred(self, text_data, batch_size=1):
        self.attn_model.eval()
        self.attn_model.to(device)

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)
        probs_all = []
        for input_ids, attn_mask, attn_select in dataloader:

            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            attn_select = attn_select.to(device)
            with torch.no_grad():
                logits, atten_score = self.attn_model(input=input_ids, labels=None,
                                                                              attn_mask=attn_mask)
                probs = nn.functional.softmax(logits, dim=-1)

                probs_all.append(probs)
        return torch.cat(probs_all, dim=0), atten_score, attn_select


class Attn_Dataset(Dataset):
    def __init__(self, map2id, max_seq_length, batch_size):

        self.map2id = map2id
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, x, map2id, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        pad_token = '<pad>'
        oov = '<oov>'
        oov_id = map2id[oov]
        features = []
        for i in range(len(x)):
            text_ls = x[i]
            # label = y[i]

            input_x_tokens = []
            input_x_ids = []
            attention_score_mask = []
            attention_score_select = []

            # label = int(label)
            length = len(text_ls)

            for index in range(len(text_ls)):
                input_x_tokens.append(text_ls[index])
                attention_score_mask.append(0)
                attention_score_select.append(1)

            if len(text_ls) > max_seq_length:
                input_x_tokens = input_x_tokens[:max_seq_length]
                attention_score_mask = attention_score_mask[:max_seq_length]
                attention_score_select = attention_score_select[:max_seq_length]

            input_x_tokens_pad = [pad_token] * (max_seq_length - length)
            attention_score_mask_pad = [float('-inf')] * (max_seq_length - length)
            attention_score_select_pad = [0] * (max_seq_length - length)

            input_x_tokens += input_x_tokens_pad
            attention_score_mask += attention_score_mask_pad
            attention_score_select += attention_score_select_pad

            for j in range(len(input_x_tokens)):
                input_x_ids.append(map2id.get(input_x_tokens[j], oov_id))

            assert len(input_x_tokens) == max_seq_length
            assert len(input_x_ids) == max_seq_length
            assert len(attention_score_mask) == max_seq_length
            assert len(attention_score_select) == max_seq_length

            features.append(
                AttnInputFeatures(input_ids=input_x_ids,
                                  # label_ids=label,
                                  attn_mask=attention_score_mask,
                                  attn_select=attention_score_select))
        return features

    def transform_text(self, text, batch_size=1):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(text, self.map2id, self.max_seq_length)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_attn_mask = torch.tensor([f.attn_mask for f in eval_features], dtype=torch.float)
        all_attn_select = torch.tensor([f.attn_select for f in eval_features], dtype=torch.bool)

        eval_data = TensorDataset(all_input_ids, all_attn_mask, all_attn_select)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


class Joint_Embedding_Model(nn.Module):
    def __init__(self, bert_embedding, glove_embedding, hidden_size, depth, dropout, nclasses=2):
        super(Joint_Embedding_Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.bert_emb_layer = modules.EmbeddingLayer(embs=bert_embedding_dataloader.load_bert_embedding_txt(bert_embedding))
        self.glove_emb_layer = modules.EmbeddingLayer(embs=bert_embedding_dataloader.load_embedding_txt(glove_embedding))
        self.bert_word2id = self.bert_emb_layer.word2id
        self.glove_word2id = self.glove_emb_layer.word2id

        self.bert_encoder = nn.LSTM(
            self.bert_emb_layer.n_d,
            hidden_size,
            depth,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
            )

        self.glove_encoder = nn.LSTM(
            self.glove_emb_layer.n_d,
            hidden_size,
            depth,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        d_out = hidden_size

        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear and sigmoid layers
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, nclasses)
        )
        self.activation = nn.Tanh()
        self.out = nn.Linear(d_out, nclasses)

    def attention_net_with_w(self, lstm_out, lstm_hidden, attn_mask):
        """
        :param lstm_out: (batch_size, seq_len, hidden_size*2)
        :param lstm_hidden: (batch_size, num_layers*num_directions, hidden_size)
        :param attn_mask: (batch_size, seq_len)
        :return: (batch_size, hidden_size)
        """

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)

        h = lstm_tmp_out[0] + lstm_tmp_out[1]

        lstm_hidden = torch.sum(lstm_hidden, dim=1)

        lstm_hidden = lstm_hidden.unsqueeze(1)

        atten_w = self.attention_layer(lstm_hidden)

        m = self.activation(h)

        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        attn_mask = attn_mask.unsqueeze(1)  #
        atten_context = torch.add(atten_context, attn_mask)
        softmax_w = F.softmax(atten_context, dim=-1)
        # print("softmax_w:", softmax_w.size(), softmax_w)

        context = torch.bmm(softmax_w, h)  # (batch_size, 1, hidden_size)
        result = context.squeeze(1)  # (batch_size, hidden_size)
        return result, softmax_w.squeeze(1)

    def forward(self, input, labels, attn_mask):
        bert_emb = self.bert_emb_layer(input)
        glove_emb = self.glove_emb_layer(input)

        bert_emb = self.drop(bert_emb)
        glove_emb = self.drop(glove_emb)

        bert_output, bert_hidden = self.bert_encoder(bert_emb)

        bert_final_hidden_state, bert_final_cell_state = bert_hidden

        glove_output, glove_hidden = self.glove_encoder(glove_emb)

        glove_final_hidden_state, glove_final_cell_state = glove_hidden

        output = bert_output + glove_output

        final_hidden_state = bert_final_hidden_state + glove_final_hidden_state

        final_hidden_state = final_hidden_state.permute(1, 0, 2)

        classify_embedding, atten_score = self.attention_net_with_w(output, final_hidden_state, attn_mask)
        classify_embedding = self.dropout(classify_embedding)
        logits = self.fc_out(classify_embedding)

        if labels is not None:

            return logits, atten_score
        else:
            return logits, atten_score


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)

    return np.sum(outputs == labels)


def eval_model(model, test_data, batch_size):
    eval_sampler = SequentialSampler(test_data)
    eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=batch_size)
    model.to(device)
    model.eval()

    with torch.no_grad():
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, label_ids, attn_mask, attn_select in eval_dataloader:

            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            attn_mask = attn_mask.to(device)
            attn_select = attn_select.to(device)

            logits, _, = model(input_ids, label_ids, attn_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids)
            # print("tmp_eval_accuracy", tmp_eval_accuracy)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)

    model.train()
    return eval_accuracy/nb_eval_examples


def train_model(epoch, model, optimizer, batch_size, num_labels, train_data, test_data, best_test, save_path):

    model.train()
    criterion = nn.CrossEntropyLoss()

    train_sampler = RandomSampler(train_data)
    # train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, label_ids, attn_mask, attn_select = batch

        optimizer.zero_grad()
        logits, attn_score = model(input_ids, label_ids, attn_mask)

        loss = criterion(logits.view(-1, num_labels), label_ids.view(-1))
        loss.backward()
        optimizer.step()

    test_acc = eval_model(model, test_data, batch_size)

    sys.stdout.write("Epoch={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(
        epoch,
        optimizer.param_groups[0]['lr'],
        loss.item(),
        test_acc
    ))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
        # test_err = eval_model(niter, model, test_x, test_y)
    sys.stdout.write("\n")
    return best_test


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', '--gpu', type=str, default="0")
    argparser.add_argument('-dataset_name', type=str, default="imdb", help="which dataset")
    argparser.add_argument('-hidden_size', type=int, default=150)
    argparser.add_argument('-depth', type=int, default=1)
    argparser.add_argument('-dropout', type=float, default=0.3)
    argparser.add_argument('-lr', type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("-batch_size", type=int, default=32)
    argparser.add_argument("-max_epoch", type=int, default=100)
    argparser.add_argument("-max_seq_length", type=int, default=128)
    args = argparser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataset_name = args.dataset_name
    hidden_size = args.hidden_size
    depth = args.depth
    dropout = args.dropout
    lr = args.lr
    lr_decay = args.lr_decay
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    max_seq_length = args.max_seq_length
    bert_embedding = 'bert_768d.txt'
    glove_embedding = 'glove.6B.200d.txt'

    if dataset_name == "imdb":
        train_x, train_y = dataloader.read_corpus('sentiment/imdb/train_tok.csv')
        test_x, test_y = dataloader.read_corpus('sentiment/imdb/test_tok.csv')

    nclasses = max(train_y) + 1
    print("nlasses:", nclasses)
    print("Number of train examples:", len(train_x))
    print("Number of test examples:", len(test_x))
    print("Check examples：")
    for i in range(len(train_y)):
        if i < 3:
            print(i)
            print(train_x[i])
            print(train_y[i])

    model = Joint_Embedding_Model(bert_embedding, glove_embedding, hidden_size, depth, dropout, nclasses).cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=lr)

    train_features = bert_embedding_dataloader.create_examples_features(train_x, train_y, model.glove_word2id, max_seq_length)
    test_features = bert_embedding_dataloader.create_examples_features(test_x, test_y, model.glove_word2id, max_seq_length)

    all_train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_train_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    all_train_attn_mask = torch.tensor([f.attn_mask for f in train_features], dtype=torch.float)
    all_train_attn_select = torch.tensor([f.attn_select for f in train_features], dtype=torch.uint8)
    train_data = TensorDataset(all_train_input_ids, all_train_label_ids, all_train_attn_mask, all_train_attn_select)

    all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_test_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)
    all_test_attn_mask = torch.tensor([f.attn_mask for f in test_features], dtype=torch.float)
    all_test_attn_select = torch.tensor([f.attn_select for f in test_features], dtype=torch.uint8)
    test_data = TensorDataset(all_test_input_ids, all_test_label_ids, all_test_attn_mask, all_test_attn_select)

    best_test = 0

    model_path = "joint_attention_models"
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    save_path = os.path.join("joint_attention_models", dataset_name + '.pt')

    for epoch in range(max_epoch):
        best_test = \
            train_model(epoch, model, optimizer, batch_size, nclasses, train_data, test_data, best_test, save_path)
        if lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= lr_decay
    sys.stdout.write("test_err: {:.6f}\n".format(
        best_test
    ))


if __name__ == "__main__":
   main()
