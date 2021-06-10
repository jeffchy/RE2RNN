import torch
from src.models.FARNN_O import IntentIntegrateOnehot
from src.data import ATISIntentBatchDataset, load_classification_dataset, load_pkl
from torch.utils.data import DataLoader
from src.utils.utils import len_stats, pad_dataset, Logger
from rules.create_logic_mat_bias import create_mat_and_bias_with_empty_TREC, create_mat_and_bias_with_empty_ATIS, create_mat_and_bias_with_empty_SMS
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from rules.fsa_to_tensor import dfa_to_tensor


def REclassifier(model, intent_dataloader, config=None, i2in=None, is_cuda=True):
    acc = 0

    model.eval()
    all_pred = []
    all_label = []
    all_out = []

    with torch.no_grad():
        for batch in intent_dataloader:

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available() and is_cuda:
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            out = model(x, lengths)

            acc += (out.argmax(1) == label).sum().item()

            all_pred += list(out.argmax(1).cpu().numpy())
            all_label += list(label.cpu().numpy())
            all_out.append(out.cpu().numpy())


    acc = acc / len(intent_dataloader.dataset)
    print('total acc: {}'.format(acc))

    if config.only_probe:
        confusion_mat = confusion_matrix(all_label, all_pred, labels=[i for i in range(config.label_size)])
        labels = [i2in[i] for i in range(config.label_size)]
        fig = plt.figure()
        fig.set_size_inches(8, 8)
        cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
        g = sns.heatmap(confusion_mat, annot=True,   cmap=cmap, linewidths=1,
                        linecolor='gray', xticklabels=labels, yticklabels=labels,)
        plt.show()

    p_micro = precision_score(all_label, all_pred, average='micro')
    r_micro = recall_score(all_label, all_pred, average='micro')
    f1_micro = f1_score(all_label, all_pred, average='micro')

    p_macro = precision_score(all_label, all_pred, average='macro')
    r_macro = recall_score(all_label, all_pred, average='macro')
    f1_macro = f1_score(all_label, all_pred, average='macro')

    # print('p_micro: {} | r_micro: {} | f1_micro: {}'.format(p_micro, r_micro, f1_micro))
    # print('p_macro: {} | r_macro: {} | f1_macro: {}'.format(p_macro, r_macro, f1_macro))

    print(f1_micro)

    return all_pred, np.concatenate(all_out)


def PredictByRE(args, params=None, dset=None,):
    logger = Logger()
    if not dset:
        dset = load_classification_dataset(args.dataset)

    t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']
    query_train, intent_train = dset['query_train'], dset['intent_train']
    query_dev, intent_dev = dset['query_dev'], dset['intent_dev']
    query_test, intent_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, train_query_inverse, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, dev_query_inverse, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, test_query_inverse, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    intent_data_train = ATISIntentBatchDataset(train_query, train_lengths, intent_train)
    intent_data_dev = ATISIntentBatchDataset(dev_query, dev_lengths, intent_dev)
    intent_data_test = ATISIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    if params is None:
        automata_dicts = load_pkl(args.automata_path_forward)
        automata = automata_dicts['automata']
        language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
        complete_tensor = language_tensor + wildcard_mat
        if args.dataset == 'ATIS':
            mat, bias = create_mat_and_bias_with_empty_ATIS(automata, in2i=in2i, i2in=i2in,)
        elif args.dataset == 'TREC':
            mat, bias = create_mat_and_bias_with_empty_TREC(automata, in2i=in2i, i2in=i2in,)
        elif args.dataset == 'SMS':
            mat, bias = create_mat_and_bias_with_empty_SMS(automata, in2i=in2i, i2in=i2in,)
    else:
        complete_tensor = params['complete_tensor']
        mat, bias = params['mat'], params['bias']

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                      config=args,
                                      mat=mat,
                                      bias=bias)

    if torch.cuda.is_available():
        model.cuda()
    # # TRAIN
    print('RE TRAIN ACC')
    all_pred_train, all_out_train = REclassifier(model, intent_dataloader_train,  config=args, i2in=i2in)
    # DEV
    print('RE DEV ACC')
    all_pred_dev, all_out_dev = REclassifier(model, intent_dataloader_dev, config=args,  i2in=i2in)
    # TEST
    print('RE TEST ACC')
    all_pred_test, all_out_test = REclassifier(model, intent_dataloader_test, config=args,i2in=i2in)

    return all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test


def PredictByRE1(args, params=None, dset=None, gpu=0):
    logger = Logger()
    device = torch.device("cuda:{}".format(gpu))

    if not dset:
        dset = load_classification_dataset(args.dataset)

    t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']
    query_train, intent_train = dset['query_train'], dset['intent_train']
    query_dev, intent_dev = dset['query_dev'], dset['intent_dev']
    query_test, intent_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, train_query_inverse, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, dev_query_inverse, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, test_query_inverse, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    intent_data_train = ATISIntentBatchDataset(train_query, train_lengths, intent_train)
    intent_data_dev = ATISIntentBatchDataset(dev_query, dev_lengths, intent_dev)
    intent_data_test = ATISIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    if params is None:
        automata_dicts = load_pkl(args.automata_path)
        automata = automata_dicts['automata']
        language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
        complete_tensor = language_tensor + wildcard_mat
        if args.dataset == 'ATIS':
            mat, bias = create_mat_and_bias_with_empty_ATIS(automata, in2i=in2i, i2in=i2in,)
        elif args.dataset == 'TREC':
            mat, bias = create_mat_and_bias_with_empty_TREC(automata, in2i=in2i, i2in=i2in,)
        elif args.dataset == 'SMS':
            mat, bias = create_mat_and_bias_with_empty_SMS(automata, in2i=in2i, i2in=i2in,)
    else:
        complete_tensor = params['complete_tensor']
        mat, bias = params['mat'], params['bias']

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                  config=args,
                                  mat=mat,
                                  bias=bias,
                                  is_cuda=False)

    # TRAIN
    print('RE TRAIN ACC')
    all_pred_train, all_out_train = REclassifier(model, intent_dataloader_train,  config=args, i2in=i2in, is_cuda=False)
    # DEV
    print('RE DEV ACC')
    all_pred_dev, all_out_dev = REclassifier(model, intent_dataloader_dev, config=args,  i2in=i2in, is_cuda=False)
    # TEST
    print('RE TEST ACC')
    all_pred_test, all_out_test= REclassifier(model, intent_dataloader_test, config=args,i2in=i2in, is_cuda=False)

    return all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0000001, help="learning rate of optimizer")
    parser.add_argument('--max_state', type=int, default=3, help="max state of each FSARNN")
    parser.add_argument('--bz', type=int, default=500, help="batch size")
    parser.add_argument('--epoch', type=int, default=200, help="max state of each FSARNN")
    parser.add_argument('--seq_max_len', type=int, default=30, help="max length of sequence")
    parser.add_argument('--early_stop', type=int, default=20, help="number of epochs that apply early stopping if dev metric not improve")
    parser.add_argument('--run', type=str, default="integrate", help="run folder name to save model")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--activation', type=str, default='relu', help='nonlinear of model, [tanh, sigmoid, relu, relu6]')
    parser.add_argument('--clamp_hidden', type=int, default=0, help='if constrain the hidden layer at each time stamp to between 0, 1')
    parser.add_argument('--l1', type=float, default=0, help="lambda for rank regularization")
    parser.add_argument('--l2', type=float, default=0., help="lambda for wildcard mat minimization")
    parser.add_argument('--train_fsa', type=int, default=1, help="if we train the fsa parameters")
    parser.add_argument('--train_wildcard', type=int, default=1, help='If we train the wildcard matrix')
    parser.add_argument('--train_linear', type=int, default=1, help="if we train the linear parameters")
    parser.add_argument('--train_V_embed', type=int, default=1, help="if we train the V_embed parameters")
    parser.add_argument('--train_word_embed', type=int, default=0, help="if we train the word embed parameters")
    parser.add_argument('--alpha', type=float, default=0.7, help="in [0,1] interval, 1 means use regex, 0 means only use embedding")
    parser.add_argument('--automata_path', type=str, default='../jupyter/rule/ATIS/AUTOMATA/150.split.pkl', help="automata path")
    parser.add_argument('--clamp_score', type=int, default=1, help='if we clamp the score in [0, 1]')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='rnn hidden dim')
    parser.add_argument('--model_type', type=str, default='FSARNN', help='baseline RNN or LSTM, or FSARNN, or Marrying Up ')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer SGD or ADAM')
    parser.add_argument('--additional_nonlinear', type=str, default='none', help='additional nonlinear after DxR, should be in [none, relu, sigmoid, tanh]')
    parser.add_argument('--additional_state', type=int, default=0, help='additional reserved state for generalization')
    parser.add_argument('--dataset', type=str, default='ATIS', help="dataset name, in ATIS, TREC, SMS")
    parser.add_argument('--regularization', type=str, default='D2', help="[V_embed, V_embed_weighted, D1, D2, None]")
    parser.add_argument('--only_probe', type=int, default=0, help='if we only prob and not train')

    args = parser.parse_args()
    all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test = PredictByRE(args)

