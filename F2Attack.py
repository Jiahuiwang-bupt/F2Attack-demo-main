import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import criteria
import pickle
import string
import csv
import copy
import math
from train_cnn_lstm_models import *
from train_bert_model import *
from train_joint_attn_model import *

from pathlib import Path
from collections import defaultdict


tf.disable_eager_execution()
use_bpe = 1  # 是否使用bpe编码
threshold_pred_score = 0.3


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        # module_url = 'https://hub.tensorflow.google.cn/google/universal-sentence-encoder-large/3'
        module_url = cache_path  # 已经下载好 可以直接用
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):
    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    # Compute the starting and ending indices of the window.
    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text
    # Calculate semantic similarity using USE.
    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
                                   list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims


def _tokenize(words, tokenizer):
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize_word_level(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + words[i + 1:])
    masked_words.append(words[0:len_text - 1])
    return masked_words


def count_perturbation_rate(x_orig, x_new):
    perturbation_num = 0
    l = len(x_orig)
    for i in range(l):
        if x_orig[i] != x_new[i]:
            perturbation_num += 1
    perturbation_rate = perturbation_num / l
    return perturbation_rate, perturbation_num


def target_function(x_orig, y_orig, x_pert, sim_score_window, sim_predictor, predictor):
    new_probs = predictor([x_pert]).squeeze()
    new_label = torch.argmax(new_probs)
    new_label = int(new_label)
    y_orig = int(y_orig)

    if new_label != y_orig:
        indicate_value = 1
    else:
        indicate_value = 0
    sim_value = calc_sim(x_orig, [x_pert], -1, sim_score_window, sim_predictor)[0]
    target_value = indicate_value * (1 - sim_value)
    return target_value


def generate_pert_text(new_state, prev_target_value, text_orig, orig_label, tgt_index, tgt_ori_word,
                       temp_synonyms_dict, predictor, sim_score_window, sim_predictor):
    prev_state = copy.deepcopy(new_state)
    pert_query_num = 0

    new_state[tgt_index] = tgt_ori_word
    new_target_value = target_function(text_orig, orig_label, new_state,
                                       sim_score_window, sim_predictor, predictor)
    pert_query_num += 1

    if new_target_value == 0:
        synonym_lis = temp_synonyms_dict[tgt_ori_word]

        tag = 0
        pert_texts = []
        candidate_texts = []
        candidate_value = []

        for i in range(len(synonym_lis)):
            new_state[tgt_index] = synonym_lis[i]

            pert_texts.append(new_state)

        for i in range(len(pert_texts)):
            value = target_function(text_orig, orig_label, pert_texts[i],
                                    sim_score_window, sim_predictor, predictor)
            pert_query_num += 1
            if value != 0:
                tag = 1
                candidate_texts.append(pert_texts[i])
                candidate_value.append(value)

        if tag == 1:
            max_value_index = candidate_value.index(min(candidate_value))

            max_value = candidate_value[max_value_index]
            best_state = candidate_texts[max_value_index]

            return pert_query_num, best_state, max_value
        else:
            return pert_query_num, prev_state, prev_target_value
    else:
        return pert_query_num, new_state, new_target_value


def cal_select_prob(sim_score_window, sim_predictor, orig_text, adv_text):
    diffs = []
    choices = []
    select_prob = []
    for i in range(len(orig_text)):
        if orig_text[i] != adv_text[i]:
            diffs.append(i)

    if len(diffs) == 0:
        return diffs, 0, 0, 0

    for j in range(len(diffs)):
        temp_replace = adv_text[:]
        temp_replace[diffs[j]] = orig_text[diffs[j]]
        temp_score = calc_sim(temp_replace, [orig_text], -1, sim_score_window, sim_predictor)[0]
        choices.append((diffs[j], temp_score))
        select_prob.append(temp_score)

    tgt_index = []

    choices.sort(key=lambda x: x[1])
    choices.reverse()
    for i in range(len(choices)):
        tgt_index.append(choices[0][0])
        break

    sec_index = random.choice(diffs)
    return diffs, tgt_index, sec_index, select_prob


def attack(text, label, allowed_qrs, pert_ratio, predictor, attn_predictor, cos_sim, word2idx, idx2word,
                  tokenizer_mlm, temp_init, max_length=128, sim_score_window=15, sim_predictor=None, synonym_num=5):
    print("text：", text)
    print("label：", label)
    k = synonym_num

    orig_probs = predictor([text]).squeeze()
    tt = predictor([text])
    orig_label = torch.argmax(orig_probs)

    label = int(label)
    label = torch.tensor(label)
    label = label.to(device)

    attn_probs, attn_scores, attn_select = attn_predictor([text])
    attn_scores = attn_scores.squeeze()
    attn_scores = torch.masked_select(attn_scores, attn_select)

    att_score = attn_scores.tolist()

    temp = att_score.strip('[').strip(']')
    temp = temp.strip("'").split("'")
    temp_1 = temp[0]
    temp_1 = temp_1.split(", ")

    attention_scores = []
    for j in range(len(temp_1)):
        temp1_1 = float(temp_1[j])
        attention_scores.append(temp1_1)
    print("attention_scores:", len(attn_scores))
    print("text:", len(text))

    if label != orig_label:
        print("Prediction results are inconsistent. Do not attack.")
        return '', 0, orig_label, orig_label, 0, 0
    else:

        words, sub_words, keys = _tokenize(text, tokenizer_mlm)

        if len(words) > max_length - 2:
            words = words[:(max_length - 2)]

        # Step 1 Two Factors Word Scoring
        # Get Attention Scores
        attn_scores = attention_scores[: (max_length - 2)]

        # Calculate Sim Scores
        words_perturb = []
        for index in range(len(words)):
            words_perturb.append((index, words[index]))

        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words = []
        synonym_values = []
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            synonym_temp = []
            for ii in res[0]:
                synonym_temp.append(ii)

            synonym_values.append(synonym_temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)

        for idx, word in words_perturb:
            candidates = []
            if word in word2idx:
                sim_synoyms = synonym_words.pop(0)
                for j in range(len(sim_synoyms)):
                    candidates.append(sim_synoyms[j])
            else:
                candidates.append(word)

            if candidates:
                candidates = candidates[:synonym_num]
                synonyms_all.append((idx, candidates))
                synonyms_dict[word] = candidates

        replace_all = []
        replace_dict = defaultdict(list)
        best_replace_scores = []
        new_attn_scores = []
        new_sim_scores = []

        for idx, word in words_perturb:
            substitutes = synonyms_dict[word]
            min_score = float("inf")
            temp_final_words = copy.deepcopy(words)
            if not substitutes:
                continue

            for substitute_ in substitutes:
                temp_replace = temp_final_words
                temp_replace[idx] = substitute_
                new_score = calc_sim(words, [temp_replace], -1, sim_score_window, sim_predictor)[0]
                if new_score < min_score:
                    best_substitute = substitute_
                    min_score = new_score

            replace_all.append((idx, best_substitute))
            replace_dict[word] = best_substitute
            dif = 1 - min_score
            if dif != 0:
                best_replace_scores.append(1 - min_score)
            else:
                if idx != len(words) - 1:
                    temp_text = temp_replace[:idx - 1] + temp_replace[idx + 1:]
                else:
                    temp_text = temp_replace[:idx - 1]
                new_score = calc_sim(words, [temp_text], -1, sim_score_window, sim_predictor)[0]
                best_replace_scores.append(new_score)
            new_attn_scores.append(attn_scores[idx])

        new_attn_scores = torch.Tensor(new_attn_scores)
        best_replace_scores = torch.Tensor(best_replace_scores)

        word_scores = torch.div(new_attn_scores, best_replace_scores)

        rank_of_index = sorted(enumerate(word_scores), key=lambda x: x[1], reverse=True)

        important_index = []
        for top_index in rank_of_index:
            tgt_word = words[top_index[0]]
            if tgt_word in string.punctuation or tgt_word in string.printable:
                continue
            else:
                important_index.append(top_index[0])

        new_words_perturb = []
        for index in range(int(pert_ratio * len(important_index))):
            new_words_perturb.append((important_index[index], words[important_index[index]]))

        # Construct synonym set for important words
        new_words_perturb_idx = [word2idx[word] for idx, word in new_words_perturb if word in word2idx]
        new_synonym_words = []
        for idx in new_words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp = []
            for ii in res[1]:
                temp.append(idx2word[ii])
            new_synonym_words.append(temp)

        new_synonyms_all = []
        new_synonyms_dict = defaultdict(list)
        temp_synonyms_dict = defaultdict(list)
        temp_synonyms_values_dict = defaultdict(list)

        for idx, word in new_words_perturb:
            candidates = []
            candidates_values = []
            if word in word2idx:
                sim_synoyms = new_synonym_words.pop(0)
                for j in range(len(sim_synoyms)):
                    if sim_synoyms[j] == word:
                        continue
                    candidates.append(sim_synoyms[j])
            else:
                candidates.append(word)
                candidates_values.append(1.0)

            if candidates:
                candidates = candidates[:synonym_num]
                temp_synonyms_dict[word] = candidates
                new_synonyms_all.append((idx, candidates))
                new_synonyms_dict[word] = candidates

        # Start Attacking!
        query_number = 0
        init_text_flag = 0
        th = 0

        # Step 2 Adversarial example initialization
        x_orig = words
        init_x_adv = x_orig[:]

        while query_number < allowed_qrs and init_text_flag == 0:
            for j in range(len(new_synonyms_all)):
                idx = new_synonyms_all[j][0]
                syn = new_synonyms_all[j][1]
                init_x_adv[idx] = random.choice(syn)
                if j >= len(x_orig):
                    break
            temp_probs = predictor([init_x_adv]).squeeze()
            temp_label = torch.argmax(temp_probs)
            query_number += 1
            if temp_label != orig_label:
                init_text_flag = 1
                break

        # Step 3 Adversarial example optimization
        query_tag = 0
        if init_text_flag == 1:
            sa_tag = 0
            text_orig = copy.deepcopy(words)
            temp_min = 0.
            temperature = temp_init
            current_state = copy.deepcopy(init_x_adv)
            best_solution = copy.deepcopy(init_x_adv)
            k = 0
            best_value = target_function(text_orig, orig_label, best_solution,
                                         sim_score_window, sim_predictor, predictor)
            query_number += 1
            prev_target_value = best_value

            prev_index = []

            while query_number <= allowed_qrs and temperature > temp_min:
                prev_state = copy.copy(current_state)
                new_state = copy.copy(current_state)

                pert_index, tgt_index, sec_index, select_prob = \
                    cal_select_prob(sim_score_window, sim_predictor, text_orig, new_state)

                if len(pert_index) == 0:
                    new_probs = predictor([best_solution]).squeeze()
                    new_label = torch.argmax(new_probs)
                    perturbation_rate, perturbation_num = count_perturbation_rate(best_solution, text_orig)
                    if new_label != label:
                        print("Find Adv Example")
                        temp_sim_1 = calc_sim(text_orig, [best_solution], -1, sim_score_window, sim_predictor)
                        final_adversarial_sim = temp_sim_1[0]
                        return ' '.join(best_solution), perturbation_num, orig_label, \
                               new_label, query_number, final_adversarial_sim

                if prev_index.count(tgt_index[0]) > 1:
                    tt_index = sec_index
                else:
                    tt_index = tgt_index[0]

                prev_index.append(tt_index)

                tgt_ori_word = text_orig[tt_index]

                pert_query_num, new_state, new_target_value = generate_pert_text(new_state, prev_target_value,
                                                                                 text_orig, orig_label, tt_index,
                                                                                 tgt_ori_word,
                                                                                 temp_synonyms_dict,
                                                                                 predictor,
                                                                                 sim_score_window, sim_predictor)

                query_number += pert_query_num
                delta_target_value = new_target_value - prev_target_value

                if new_target_value != 0:
                    if delta_target_value <= 0:
                        current_state = new_state
                        prev_target_value = new_target_value
                        if new_target_value < best_value:
                            best_solution = new_state
                            best_value = new_target_value
                            continue
                    else:
                        transfer_prob = math.exp(-delta_target_value / temperature)
                        if np.random.rand() < transfer_prob:
                            current_state = new_state
                            prev_target_value = new_target_value
                            if prev_target_value < best_value:
                                best_solution = current_state
                                best_value = prev_target_value
                                continue

                    temperature = temperature * 0.9
                    k = k + 1
                else:
                    sa_tag = 1
                    break

            if sa_tag == 1:
                new_probs = predictor([prev_state]).squeeze()
                new_label = torch.argmax(new_probs)
                perturbation_rate, perturbation_num = count_perturbation_rate(prev_state, text_orig)
                if new_label != label:
                    temp_sim_1 = calc_sim(text_orig, [prev_state], -1, sim_score_window, sim_predictor)
                    final_adversarial_sim = temp_sim_1[0]
                    return ' '.join(init_x_adv), perturbation_num, orig_label, \
                           new_label, query_number, final_adversarial_sim
                else:
                    new_probs = predictor([init_x_adv]).squeeze()
                    new_label = torch.argmax(new_probs)
                    perturbation_rate, perturbation_num = count_perturbation_rate(init_x_adv, text_orig)
                    temp_sim_1 = calc_sim(text_orig, [init_x_adv], -1, sim_score_window, sim_predictor)
                    final_adversarial_sim = temp_sim_1[0]
                    return ' '.join(init_x_adv), perturbation_num, orig_label, \
                           new_label, query_number, final_adversarial_sim
            else:
                new_probs = predictor([best_solution]).squeeze()
                new_label = torch.argmax(new_probs)
                perturbation_rate, perturbation_num = count_perturbation_rate(best_solution, text_orig)
                temp_sim_1 = calc_sim(text_orig, [best_solution], -1, sim_score_window, sim_predictor)
                final_adversarial_sim = temp_sim_1[0]
                return ' '.join(best_solution), perturbation_num, orig_label, \
                       new_label, query_number, final_adversarial_sim
        elif init_text_flag == 1 and query_tag == 1:
            new_probs = predictor([init_x_adv]).squeeze()
            new_label = torch.argmax(new_probs)
            perturbation_rate, perturbation_num = count_perturbation_rate(init_x_adv, words)
            temp_sim_1 = calc_sim(words, [init_x_adv], -1, sim_score_window, sim_predictor)
            final_adversarial_sim = temp_sim_1[0]
            return ' '.join(init_x_adv), perturbation_num, orig_label, \
                   new_label, query_number, final_adversarial_sim
        else:
            return '', 0, orig_label, orig_label, 0, 0


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', '--gpu', type=str, default="0")
    argparser.add_argument('-model_type', type=str, default="wordCNN")
    argparser.add_argument('-attention_dataset_name', type=str, default="imdb", help="which attention dataset")
    argparser.add_argument('-target_dataset_name', type=str, default="mr", help="which target dataset")
    argparser.add_argument('-num_labels', type=int, default=2)
    argparser.add_argument('-hidden_size', type=int, default=150)
    argparser.add_argument('-depth', type=int, default=1)
    argparser.add_argument('-dropout', type=float, default=0.)
    argparser.add_argument('-pert_ratio', type=float, default=0.7)
    argparser.add_argument("-batch_size", type=int, default=32)
    argparser.add_argument("-temp_init", type=int, default=30)
    argparser.add_argument("-max_seq_length", type=int, default=128)
    argparser.add_argument("-data_size", type=int, default=500)
    argparser.add_argument("-synonym_num", type=int, default=5)
    argparser.add_argument("-allowed_qrs", type=int, default=100)
    argparser.add_argument("-USE_cache_path", type=str, default="use_model")

    args = argparser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model_type = args.model_type
    attention_dataset_name = args.attention_dataset_name
    target_dataset_name = args.target_dataset_name
    num_labels = args.num_labels
    hidden_size = args.hidden_size
    temp_init = args.temp_init
    attention_model_version = args.attention_model_version
    allowed_qrs = args.allowed_qrs
    synonym_num = args.synonym_num
    data_size = args.data_size
    max_seq_length = args.max_seq_length
    USE_cache_path = args.USE_cache_path
    pert_ratio = args.pert_ratio
    bert_embedding = 'bert_768d.txt'
    glove_embedding = 'glove.6B.200d.txt'

    mlm_model_dir = "bert-base-uncased"
    use = USE(USE_cache_path)

    print("Loading Target Model")
    if model_type == "wordCNN":
        cnn = True
        target_model_path = os.path.join("cnn_models", target_dataset_name + ".pt")

        model = CNN_Lstm_Model(glove_embedding, nclasses=num_labels, hidden_size=150, cnn=cnn).cuda()
        checkpoint = torch.load(target_model_path, map_location="cuda:" + args.gpu)
        model.load_state_dict(checkpoint)
    elif model_type == "wordLSTM":
        cnn = False
        target_model_path = os.path.join("lstm_models", target_dataset_name + "_lstm.pt")

        model = CNN_Lstm_Model(glove_embedding, nclasses=num_labels, hidden_size=150, cnn=cnn).cuda()
        checkpoint = torch.load(target_model_path, map_location="cuda:" + args.gpu)
        model.load_state_dict(checkpoint)
    elif model_type == "bert":
        target_model_path = "models_" + target_dataset_name
        model = NLI_BERT(target_model_path, nclasses=num_labels, max_seq_length=max_seq_length)

    predictor = model.text_pred
    print("Target Model built!")

    print("Loading Masked Language Model")
    tokenizer_mlm = BertTokenizer.from_pretrained(mlm_model_dir, do_lower_case=True)
    print("Masked Language Model built!")

    print("Loading Counter-fitted Word Embedding Space")
    counter_fitting_embeddings_path = "counter-fitted-vectors.txt"
    counter_fitting_cos_sim_path = "mat.txt"
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1
    print("The size of vocab：", len(idx2word))

    print("Building cos sim matrix...")
    if counter_fitting_cos_sim_path:
        print('Load pre-computed cosine similarity matrix from {}'.format(counter_fitting_cos_sim_path))
        with open(counter_fitting_cos_sim_path, "rb+") as fp:
            cos_sim = pickle.load(fp)
    else:
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Counter-fitted Word Embedding Space Built!")

    if target_dataset_name == "mr":
        test_x, test_y = dataloader.read_corpus('sentiment/mr/test.txt')
    elif target_dataset_name == "imdb":
        test_x, test_y = dataloader.read_corpus('sentiment/imdb/test_tok.csv')
    elif target_dataset_name == "yahoo":
        test_x, test_y = dataloader.read_yahoo('yahoo_answers_csv/test.csv')
    elif target_dataset_name == "sst2":
        test_x, test_y = dataloader.read_sst2_corpus('sentiment/sst2/test.tsv')
    elif target_dataset_name == "jigsaw":
        test_x, test_y = dataloader.read_security_corpus('security/hatespeech/jigsaw/dev.csv')
    elif target_dataset_name == "HSOL":
        test_x, test_y = dataloader.read_security_corpus('security/hatespeech/HSOL/dev.csv')
    elif target_dataset_name == "FAS":
        test_x, test_y = dataloader.read_security_corpus('security/sensitive/FAS/dev.csv')
    else:
        test_x, test_y = dataloader.read_security_corpus('security/sensitive/EDENCE/dev.csv')

    test_x = test_x[:data_size]
    test_y = test_y[:data_size]

    data = list(zip(test_x, test_y))
    attack_data_size = len(data)
    print("Attack size：", attack_data_size)
    attention_model_dir = os.path.join("joint_attention_models", attention_dataset_name + ".pt")
    attn_model = Attn_Model(attention_model_dir, bert_embedding, glove_embedding, max_seq_length=max_seq_length,
                            hidden_size=hidden_size, depth=args.depth, dropout=args.dropout, nclasses=num_labels)
    attn_predictor = attn_model.text_pred
    print("Attention Model Built!")

    print("Start Attacking!")
    orig_failures = 0.
    adv_failures = 0.
    ori_classify_correct = []
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    f_queries = []
    results = []
    final_sims = []
    random_changed_rates = []
    model_predict_true_attack_suc = 0
    model_predict_true_attack_fai = 0

    log_dir = "F2Attack" + "/" + str(args.allowed_qrs) + "_queries" + "/" + \
              str(args.data_size) + "_examples" + "/" + model_type + "/" + target_dataset_name

    res_dir = "F2Attack" + "/" + str(args.allowed_qrs) + "_queries" + "/" + \
              str(args.data_size) + "_examples" + "/" + model_type + "/" + target_dataset_name

    log_file = "F2Attack" + "/" + str(args.allowed_qrs) + "_queries" + "/" + \
               str(args.data_size) + "_examples" + "/" + model_type + "/" \
               + target_dataset_name + "/attack_sim_lis" + ".txt"

    result_file = "F2Attack" + "/" + str(args.allowed_qrs) + "_queries" + "/" + \
                  str(args.data_size) + "_examples" + "/" + model_type + "/" \
                  + target_dataset_name + "/results_sim_lis_back" + ".csv"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    output_dir = os.path.join("F2Attack_result",
                              str(args.allowed_qrs) + "_queries",
                              str(args.data_size) + "_examples",
                              model_type, target_dataset_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if attention_model_version == 3:
        for idx, (text, label) in enumerate(data):
            if idx % 20 == 0:
                print(str(idx) + " Samples Done")
            new_text, num_changed, orig_label, new_label, model_num_queries, sim = \
                attack(text, label, allowed_qrs, pert_ratio, predictor, attn_predictor,
                       cos_sim, word2idx, idx2word, tokenizer_mlm, temp_init,
                       max_length=max_seq_length, sim_score_window=15, sim_predictor=use,
                       synonym_num=synonym_num)
            label = int(label)
            orig_label = int(orig_label)
            new_label = int(new_label)

            if label != orig_label:
                orig_failures += 1
            else:
                ori_classify_correct.append(1)

            if label != new_label:
                adv_failures += 1

            # split_text = text.split()
            changed_rate = 1.0 * num_changed / len(text)

            if label == orig_label and label != new_label:
                temp = []
                model_predict_true_attack_suc = model_predict_true_attack_suc + 1
                changed_rates.append(changed_rate)
                orig_texts.append(' '.join(text))
                adv_texts.append(new_text)
                true_labels.append(label)
                new_labels.append(new_label)
                nums_queries.append(model_num_queries)
                final_sims.append(sim)
                temp.append(idx)
                temp.append(orig_label)
                temp.append(new_label)
                temp.append(' '.join(text))
                temp.append(new_text)
                temp.append(model_num_queries)
                temp.append(sim)
                temp.append(changed_rate * 100)
                results.append(temp)
                print("Attacked: " + str(idx))

            if label == orig_label and label == new_label:
                model_predict_true_attack_fai = model_predict_true_attack_fai + 1
                f_queries.append(model_num_queries)

        with open(result_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(results)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, attention_dataset_name + "_" + target_dataset_name + "_" + "sim_lis_back.txt"),
                  'w') as ofile:
            for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
                ofile.write(
                    'orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

        message = 'For target model  {} on {} dataset ' '' \
                  'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
                  'avg changed rate: {:.3f}%, num of queries: {:.1f}, ' \
                  'final_sims : {:.3f}%, attack_success_rate : {:.3f}%\n'. \
            format(model_type,
                   target_dataset_name,
                   (1 - orig_failures / attack_data_size) * 100,
                   (1 - adv_failures / attack_data_size) * 100,
                   np.mean(random_changed_rates) * 100,
                   np.mean(changed_rates) * 100,
                   np.mean(nums_queries),
                   np.mean(final_sims),
                   (model_predict_true_attack_suc / len(ori_classify_correct)) * 100)
        log = open(log_file, 'a')
        log.write(message)
        print(message)
        print(orig_failures)
        print(adv_failures)


if __name__ == "__main__":
    main()

