from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from collections import Counter
from rules.load_data_and_rules import load_TREC_dataset, load_SMS_dataset, load_rule
from pydash.arrays import compact
from automata_tools import NFAtoDFA, DFAtoMinimizedDFA
from rules.dfa_from_rule import NFAFromRegex
from rules.fsa_to_tensor import Automata, drawGraph, dfa_to_tensor
from src.utils.utils import mkdir, evan_select_from_total_number
from rules.tensor_func import tensor3_to_factors
import time
import argparse

class ATISIntentDataset(Dataset):

    def __init__(self, query, intent):
        assert len(query) == len(intent)
        self.dataset = query
        self.intent = intent

    def __getitem__(self, idx):
        return {
            'x':self.dataset[idx],
            'i':self.intent[idx]
        }

    def __len__(self):
        return len(self.dataset)

class ATISIntentBatchDataset(Dataset):

    def __init__(self, query, lengths, intent, shots=None):
        assert len(query) == len(intent)
        if (not shots) or (shots > len(query)):
            self.dataset = query
            self.intent = intent
            self.lengths = lengths
        else:
            # idxs = np.random.choice(np.arange(len(query)), size=int(portion* len(query)), replace=False)
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)


class ATISIntentBatchDatasetBidirection(Dataset):

    def __init__(self, query, query_inverse, lengths, intent, shots=None):
        assert len(query) == len(intent)
        if (not shots) or (shots > len(query)):
            self.dataset = query
            self.dataset_inverse = query_inverse
            self.intent = intent
            self.lengths = lengths
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.dataset_inverse = list(np.array(query_inverse)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])

    def __getitem__(self, idx):

        return {
            'x_forward': np.array(self.dataset[idx]),
            'x_backward': np.array(self.dataset_inverse[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)

class ATISIntentBatchDatasetUtilizeUnlabel(Dataset):

    def __init__(self, query, query_inverse, lengths, intent_gold, intent_re,  re_out, shots=None):
        assert len(query) == len(intent_gold)
        assert len(query) == len(intent_re)
        self.dataset = query
        self.lengths = lengths
        self.re_out = re_out
        self.dataset_inverse = query_inverse

        if (shots == None) or (shots > len(query)):
            self.intent = intent_gold
        elif shots == 0:
            self.intent = intent_re
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            new_intent = np.array(intent_re)
            selected = np.array(intent_gold)[idxs].reshape(-1)
            new_intent[idxs] = selected
            self.intent = list(new_intent)

    def __getitem__(self, idx):

        return {
            'x_forward': np.array(self.dataset[idx]),
            'x_backward': np.array(self.dataset_inverse[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)


class MarryUpIntentBatchDataset(Dataset):

    def __init__(self, query, lengths, intent, re_out, shots=None):
        assert len(query) == len(intent)
        if (shots == None) or (shots > len(query)):
            self.dataset = query
            self.intent = intent
            self.lengths = lengths
            self.re_out = re_out
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])
            self.re_out = list(np.array(re_out)[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx]),
            're': np.array(self.re_out[idx])
        }

    def __len__(self):
        return len(self.dataset)

class MarryUpIntentBatchDatasetUtilizeUnlabel(Dataset):

    def __init__(self, query, lengths, intent_gold, intent_re,  re_out, shots=None):
        assert len(query) == len(intent_gold)
        assert len(query) == len(intent_re)
        self.dataset = query
        self.lengths = lengths
        self.re_out = re_out
        if (shots == None) or (shots > len(query)):
            self.intent = intent_gold
        elif shots == 0:
            self.intent = intent_re
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            new_intent = np.array(intent_re)
            selected = np.array(intent_gold)[idxs].reshape(-1)
            new_intent[idxs] = selected
            self.intent = list(new_intent)

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx]),
            're': np.array(self.re_out[idx])
        }

    def __len__(self):
        return len(self.dataset)


def make_glove_embed(glove_path, dataset_path, i2t, embed_dim='100'):
    glove = {}
    vecs = [] # use to produce unk

    # load glove
    with open(os.path.join(glove_path, 'glove.6B.{}d.txt'.format(embed_dim)),
              'r', encoding='utf-8') as f:
        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            embed_float = [float(i) for i in embed_str]
            if word not in glove:
                glove[word] = embed_float
                vecs.append(embed_float)
    unk = np.mean(vecs, axis=0)

    # load glove to task vocab
    embed = []
    for i in i2t:
        word = i2t[i].lower()
        if word in glove:
            embed.append(glove[word])
        else:
            embed.append(unk)

    final_embed = np.array(embed, dtype=np.float)
    pickle.dump(final_embed, open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'wb'))

    return


def load_glove_embed(dataset_path, embed_dim):
    """
    :param config:
    :return the numpy array of embedding of task vocabulary: V x D:
    """
    return pickle.load(open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'rb'))


def load_ds(fname='../data/ATIS/atis.train.pkl'):
    print(os.listdir())
    with open(fname,'rb') as stream:
        ds, dicts = pickle.load(stream)
    print('Done  loading: ', fname)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds, dicts


def read_ATIS(mode='train', DATA_DIR = '../data/ATIS'):
    # function to read ATIS dataset, calling example:
    # t2i, s2i, in2i, i2t, i2s, i2in, query, slots, intent = read_ATIS(mode='train')

    train_ds, dicts = load_ds(os.path.join(DATA_DIR, 'atis.{}.pkl'.format(mode)))
    test_ds, _ = load_ds(os.path.join(DATA_DIR, 'atis.test.pkl'))
    assert dicts == _

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
    query, slots, intent = map(train_ds.get,
                               ['query', 'slot_labels', 'intent_labels'])

    return t2i, s2i, in2i, i2t, i2s, i2in, query, slots, intent, dicts


def create_ATIS_toy_dataset(DATA_DIR = '../data/ATIS'):

    train_ds, dicts = load_ds(os.path.join(DATA_DIR, 'atis.{}.pkl'.format('train')))

    # create a smaller dataset for development
    samples = 20
    sampled_train_ds = {}
    sampled_train_ds['slot_labels'] = train_ds['slot_labels'][:samples]
    sampled_train_ds['query'] = train_ds['query'][:samples]
    sampled_train_ds['intent_labels'] = train_ds['intent_labels'][:samples]
    pickle.dump((sampled_train_ds, dicts), open('../data/atis.toy.{}.train.pkl'.format(samples), 'wb'))


def load_pkl(path):
    print(path)
    dicts = pickle.load(open(path, 'rb'))
    return dicts


def load_dict_ATIS(DATA_DIR = '../data/ATIS'):

    t2i, s2i, in2i, i2t, i2s, i2in, dicts = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format('dicts')), 'rb'))
    return t2i, i2t, in2i, i2in, dicts


def load_data_ATIS(mode, DATA_DIR = '../data/ATIS'):

    query, slots, intent = pickle.load(open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format(mode)), 'rb'))
    return query, slots, intent


def save_data(query, slots, intent, mode, DATA_DIR = '../data/ATIS'):
    pickle.dump((query, slots, intent), open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format(mode)), 'wb'))

def train_split(query, slots, intent, rate=0.8, DATA_DIR = '../data/ATIS'):


    lengths = len(query)
    assert lengths == len(slots)
    shuffle_idx = np.arange(lengths)
    np.random.shuffle(shuffle_idx)
    query = np.array(query)
    slots = np.array(slots)
    intent = np.array(intent)

    query = query[shuffle_idx]
    slots = slots[shuffle_idx]
    intent = intent[shuffle_idx]
    train_query = query[: int(lengths*rate)]
    train_slots = slots[: int(lengths*rate)]
    train_intent = intent[: int(lengths*rate)]
    dev_query = query[int(lengths*rate):]
    dev_slots = slots[int(lengths*rate):]
    dev_intent = intent[int(lengths*rate):]
    pickle.dump((train_query, train_slots, train_intent), open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format('train')), 'wb'))
    pickle.dump((dev_query, dev_slots, dev_intent), open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format('dev')), 'wb'))


def create_vocabs(iterable, mode):
    assert mode in ['labels', 'texts']
    vocab = Counter()
    if mode == 'labels':
        vocab = vocab + Counter(list(iterable))
    else:
        for instance in iterable:
            vocab += Counter(instance)

    vocab_list = list(vocab.keys())
    i2v = {idx: vocab for idx, vocab in enumerate(vocab_list)}
    v2i = {vocab: idx for idx, vocab in enumerate(vocab_list)}

    return i2v, v2i


def load_classification_dataset(dataset, datadir='../data/'):
    assert dataset in ['ATIS', 'MITR', 'SMS', 'SMS0.8', 'SMS0.5', 'SMS0.3', 'SMS0.1', 'TREC', 'Youtube', ]
    if dataset == 'ATIS':
        t2i, i2t, in2i, i2in, dicts = load_dict_ATIS(DATA_DIR='{}/ATIS'.format(datadir))
        query_train, slots_train, intent_train = load_data_ATIS(mode='train', DATA_DIR='{}/ATIS'.format(datadir))
        query_dev, slots_dev, intent_dev = load_data_ATIS(mode='dev',  DATA_DIR='{}/ATIS'.format(datadir))
        query_test, slots_test, intent_test = load_data_ATIS(mode='test',  DATA_DIR='{}/ATIS'.format(datadir))
        return {
            't2i': t2i, 'i2t': i2t, 'in2i': in2i, 'i2in': i2in,
            'query_train': query_train, 'intent_train': intent_train,
            'query_dev': query_dev, 'intent_dev': intent_dev,
            'query_test': query_test, 'intent_test': intent_test,
        }

    else:
        if 'SMS' in dataset:
            dataset = 'SMS'
        return pickle.load(open('{}{}/dataset.pkl'.format(datadir, dataset), 'rb'))


def create_classification_dataset(dataset_name):
    if dataset_name == 'TREC':
        res = load_TREC_dataset()
    elif dataset_name == 'SMS':
        res = load_SMS_dataset()
    elif dataset_name == 'ATIS':
        print("LOADING ATIS DATASET")
        res = load_classification_dataset('ATIS')
        print('CREATE EMBED FILE')
        make_glove_embed('../data/emb/glove.6B', '../data/{}'.format(dataset_name), res['i2t'])
        print('SAVING DATASET')
        pickle.dump(res, open('../data/{}/dataset.pkl'.format(dataset_name), 'wb'))
        return
    else:
        raise ValueError('WRONG DATASET NAME')

    print('CREATING VOCAB FILES')
    data, rules = res['data'], res['rules']
    labels = list(data['class'])
    texts = list(data['text'])
    texts = [['BOS'] + i.strip().split() + ['EOS'] for i in texts]
    i2in, in2i  = create_vocabs(labels, 'labels')
    i2t, t2i = create_vocabs(texts, 'texts')

    print('CREATING EMBED FILE')
    make_glove_embed('../data/emb/glove.6B', '../data/{}'.format(dataset_name), i2t)

    print('TRANSFORMING TO INDEX')
    data = data.groupby('mode')
    train, dev, test = data.get_group('train'), data.get_group('valid'), data.get_group('test')

    def to_query_intent(dataset):
        labels = dataset['class']
        texts = dataset['text']
        texts = [['BOS'] + i.strip().split() + ['EOS'] for i in texts]
        intent = [in2i[i] for i in labels]
        query = [[t2i[j] for j in i] for i in texts]
        return intent, query


    intent_train, query_train = to_query_intent(train)
    intent_dev, query_dev = to_query_intent(dev)
    intent_test, query_test = to_query_intent(test)

    print('SAVING DATASET')
    dataset = {
        't2i': t2i, 'i2t': i2t, 'in2i': in2i, 'i2in': i2in,
        'query_train': query_train, 'intent_train': intent_train,
        'query_dev': query_dev, 'intent_dev': intent_dev,
        'query_test': query_test, 'intent_test': intent_test,
    }
    pickle.dump(dataset, open('../data/{}/dataset.pkl'.format(dataset_name), 'wb'))


def decompose_tensor_split(
    language_tensor, language, word2idx, rank, random_state=1, n_iter_max=100, init='svd'
):
    language_tensor_squashed = language_tensor[np.array([word2idx[i] for i in language])]

    print('SQUASHED TENSOR SIZE: {}'.format(language_tensor_squashed.shape))
    time_start = time.time()
    V_split, D1_split, D2_split, rec_error = tensor3_to_factors(language_tensor_squashed, rank=rank,
                                                     n_iter_max=n_iter_max, init=init, verbose=10, random_state=random_state)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    Vocab, State, _ = language_tensor.shape
    V_embed_split = np.zeros((Vocab, rank))
    for i in range(len(language)):
        idx = word2idx[language[i]]
        V_embed_split[idx] = V_split[i]

    return V_embed_split, D1_split, D2_split, rec_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ATIS', help="dataset name, ATIS, TREC, SMS")
    args = parser.parse_args()
    assert args.dataset in ['ATIS', 'TREC', 'SMS']

    create_classification_dataset(args.dataset)
