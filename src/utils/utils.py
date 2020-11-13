import numpy as np
import random
import torch
import os
import datetime, time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def len_stats(query):
    max_len = 0
    avg_len = 0
    for q in query:
        max_len = max(len(q), max_len)
        avg_len += len(q)
    avg_len /= len(query)

    print("max_len: {}, avg_len: {}".format(max_len, avg_len))

def pad_dataset(query, config, pad_idx):

    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)
        q_inverse = q[::-1]

        if length > config.seq_max_len:
            q = q[: config.seq_max_len]
            q_inverse = q_inverse[: config.seq_max_len]
            length = config.seq_max_len
        else:
            remain = config.seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == config.seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths


def pad_dataset_1(query, seq_max_len, pad_idx):

    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)

        if length <= 0:
            continue

        q_inverse = q[::-1]

        if length > seq_max_len:
            q = q[: seq_max_len]
            q_inverse = q_inverse[: seq_max_len]
            length = seq_max_len
        else:
            remain = seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m%d%H%M%S")
    datetime_str = datetime_str + '-' + str(time.time())
    return datetime_str

class Args():
    def __init__(self, data):
        self.data = data
        for k, v in data.items():
            setattr(self, k, v)

class Logger():
    def __init__(self):
        self.record = [] # recored strings

    def add(self, string):
        assert type(string) == str
        self.record.append(string+' \n')

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(self.record)


def get_automata_from_seed(args, seed):

    dset = args.dataset
    if 'SMS' in dset:
        dset = 'SMS'

    if args.automata_path_forward != 'none':
        return '../data/{}/automata/{}'.format(dset, args.automata_path_forward), \
           '../data/{}/automata/{}'.format(dset, args.automata_path_backward)

    assert seed in [0,1,2,3]
    if args.dataset == 'ATIS':

        automata_list = [
            'automata.newrule.split.randomseed150.False.0.0003.0.pkl',
            'automata.newrule.split.randomseed150.False.0.0003.1.pkl',
            'automata.newrule.split.randomseed150.False.0.0007.2.pkl',
            'automata.newrule.split.randomseed150.False.0.0741.3.pkl'
        ]

        automata_list_reverse = [
            'automata.newrule.reversed.randomseed150.False.0.0735.0.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0006.1.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0735.2.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0735.3.pkl',
        ]

    elif args.dataset == 'TREC':
        automata_list = [
            'automata.split.randomseed.200.False.0.0021.0.pkl',
            'automata.split.randomseed.200.False.0.0018.1.pkl',
            'automata.split.randomseed.200.False.0.0018.2.pkl',
            'automata.split.randomseed.200.False.0.0021.3.pkl',
        ]

        automata_list_reverse = [
            'automata.newrule.reversed.randomseed200.False.0.0354.0.pkl',
            'automata.newrule.reversed.randomseed200.False.0.0354.1.pkl',
            'automata.newrule.reversed.randomseed200.False.0.0501.2.pkl',
            'automata.newrule.reversed.randomseed200.False.0.0354.3.pkl',
        ]

    # elif args.dataset == 'TREC-hotswap':
    #     automata_list = [
    #         'automata.hotswap.split.randomseed150.False.0.0064.0.pkl',
    #         'automata.hotswap.split.randomseed150.False.0.0089.1.pkl',
    #         'automata.hotswap.split.randomseed150.False.0.0068.2.pkl',
    #         'automata.hotswap.split.randomseed150.False.0.0086.3.pkl',
    #     ]
    #
    #     automata_list_reverse = [
    #         'automata.hotswap.reversed.randomseed150.False.0.0413.0.pkl',
    #         'automata.hotswap.reversed.randomseed150.False.0.0414.1.pkl',
    #         'automata.hotswap.reversed.randomseed150.False.0.0412.2.pkl',
    #         'automata.hotswap.reversed.randomseed150.False.0.0414.3.pkl',
    #     ]
    #     args.dataset = 'TREC'



    elif args.dataset  == 'SMS':
        automata_list = [
            'automata.split.randomseed.150.False.0.0002.0.pkl',
            'automata.split.randomseed.150.False.0.0002.1.pkl',
            'automata.split.randomseed.150.False.0.0003.2.pkl',
            'automata.split.randomseed.150.False.0.0004.3.pkl',
        ]

        automata_list_reverse = [
            'automata.newrule.reversed.randomseed150.False.0.0003.0.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0003.1.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0001.2.pkl',
            'automata.newrule.reversed.randomseed150.False.0.0001.3.pkl'
        ]

    # elif args.dataset  == 'SMS0.8':
    #     automata_list = [
    #         'automata.SMS0.8.split.randomseed150.False.0.0003.0.pkl',
    #         'automata.SMS0.8.split.randomseed150.False.0.0003.1.pkl',
    #         'automata.SMS0.8.split.randomseed150.False.0.0004.2.pkl',
    #         'automata.SMS0.8.split.randomseed150.False.0.0003.3.pkl',
    #     ]
    #
    #     automata_list_reverse = [
    #         'automata.SMS0.8.reversed.randomseed150.False.0.0006.0.pkl',
    #         'automata.SMS0.8.reversed.randomseed150.False.0.0002.1.pkl',
    #         'automata.SMS0.8.reversed.randomseed150.False.0.0003.2.pkl',
    #         'automata.SMS0.8.reversed.randomseed150.False.0.0006.3.pkl'
    #     ]
    #     args.additional_state = 0
    #
    # elif args.dataset  == 'SMS0.5':
    #     automata_list = [
    #         'automata.SMS0.5.split.randomseed150.False.0.0005.0.pkl',
    #         'automata.SMS0.5.split.randomseed150.False.0.0007.1.pkl',
    #         'automata.SMS0.5.split.randomseed150.False.0.0004.2.pkl',
    #         'automata.SMS0.5.split.randomseed150.False.0.0005.3.pkl',
    #     ]
    #
    #     automata_list_reverse = [
    #         'automata.SMS0.5.reversed.randomseed150.False.0.0004.0.pkl',
    #         'automata.SMS0.5.reversed.randomseed150.False.0.0003.1.pkl',
    #         'automata.SMS0.5.reversed.randomseed150.False.0.0005.2.pkl',
    #         'automata.SMS0.5.reversed.randomseed150.False.0.0005.3.pkl'
    #     ]
    #     args.additional_state = 0
    #
    # elif args.dataset  == 'SMS0.3':
    #     automata_list = [
    #         'automata.SMS0.3.split.randomseed150.False.0.0014.0.pkl',
    #         'automata.SMS0.3.split.randomseed150.False.0.0013.1.pkl',
    #         'automata.SMS0.3.split.randomseed150.False.0.0017.2.pkl',
    #         'automata.SMS0.3.split.randomseed150.False.0.0017.3.pkl',
    #     ]
    #
    #     automata_list_reverse = [
    #         'automata.SMS0.3.reversed.randomseed150.False.0.0015.0.pkl',
    #         'automata.SMS0.3.reversed.randomseed150.False.0.0013.1.pkl',
    #         'automata.SMS0.3.reversed.randomseed150.False.0.0013.2.pkl',
    #         'automata.SMS0.3.reversed.randomseed150.False.0.0013.3.pkl'
    #     ]
    #     args.additional_state = 0
    #
    #
    # elif args.dataset  == 'SMS0.1':
    #     automata_list = [
    #         'automata.SMS0.1.split.randomseed150.False.0.0000.0.pkl',
    #         'automata.SMS0.1.split.randomseed150.False.0.0000.1.pkl',
    #         'automata.SMS0.1.split.randomseed150.False.0.0000.2.pkl',
    #         'automata.SMS0.1.split.randomseed150.False.0.0000.3.pkl',
    #     ]
    #
    #     automata_list_reverse = [
    #         'automata.SMS0.1.reversed.randomseed150.False.0.0000.0.pkl',
    #         'automata.SMS0.1.reversed.randomseed150.False.0.0000.1.pkl',
    #         'automata.SMS0.1.reversed.randomseed150.False.0.0000.2.pkl',
    #         'automata.SMS0.1.reversed.randomseed150.False.0.0000.3.pkl'
    #     ]
    #     args.additional_state = 0

    return '../data/{}/automata/{}'.format(dset, automata_list[seed]), \
           '../data/{}/automata/{}'.format(dset, automata_list_reverse[seed])



def relu_normalized_NLLLoss(input, target):
    relu = torch.nn.ReLU()
    loss = torch.nn.NLLLoss()
    input = relu(input)
    input += 1e-4 # add small positive offset to prevent 0
    input = input / torch.sum(input)
    input = torch.log(input)
    loss_val = loss(input, target)
    return loss_val


def even_select_from_portion(L, portion):
    final_nums = int(L * portion)
    interval = 1 / portion
    idxs = [int(i*interval) for i in range(final_nums)]
    return np.array(idxs)

def evan_select_from_total_number(L, N):
    assert L >= N
    if N > 0:
        portion = N / L
        interval = 1 / portion
        idxs = [int(i*interval) for i in range(N)]
    else:
        idxs = []
    return np.array(idxs)




