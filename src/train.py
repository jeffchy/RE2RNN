import torch
from src.models.FARNN import IntentIntegrateSaperate_B, IntentIntegrateSaperateBidirection_B,\
    FSARNNIntegrateEmptyStateSaperateGRU
from src.models.FARNN_O import IntentIntegrateOnehot
from src.models.Baseline import IntentMarryUp
from src.data import ATISIntentBatchDataset, ATISIntentBatchDatasetBidirection, ATISIntentBatchDatasetUtilizeUnlabel, MarryUpIntentBatchDataset, \
    load_glove_embed, load_classification_dataset, load_pkl, MarryUpIntentBatchDatasetUtilizeUnlabel
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.utils import len_stats, pad_dataset, mkdir, create_datetime_str, Logger, relu_normalized_NLLLoss
from rules.create_logic_mat_bias import create_mat_and_bias_with_empty_TREC, create_mat_and_bias_with_empty_ATIS, create_mat_and_bias_with_empty_SMS
import numpy as np
from src.RE import PredictByRE
from rules.fsa_to_tensor import dfa_to_tensor
import pickle
from src.val import val, val_marry
from copy import deepcopy


def get_average(M, normalize_type):
    """
    :param M:
    :param normalize_type:
    :return:
    Get the averaged norm
    """
    assert normalize_type in ['l1', 'l2']
    Dims = M.shape
    eles = 1
    for D in Dims:
        eles *= D
    if normalize_type == 'l1':
        temp = np.linalg.norm(M, 1)
    elif normalize_type == 'l2':
        temp = np.linalg.norm(M, 2)

    return temp / eles


def get_init_params(config, in2i, i2in, t2i, automata_path):

    dset = config.dataset
    if 'SMS' in config.dataset:
        dset = 'SMS'

    pretrained_embed = load_glove_embed('../data/{}/'.format(dset), config.embed_dim)
    if config.random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)
    automata_dicts = load_pkl(automata_path)
    automata = automata_dicts['automata']

    V_embed, D1, D2 = automata_dicts['V'], automata_dicts['D1'], automata_dicts['D2']
    wildcard_mat, language = automata_dicts['wildcard_mat'], automata_dicts['language']

    n_vocab, rank = V_embed.shape
    n_state, _ = D1.shape
    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape
    if dset == 'ATIS':
        mat, bias = create_mat_and_bias_with_empty_ATIS(automata, in2i=in2i, i2in=i2in,)
    elif dset== 'TREC':
        mat, bias = create_mat_and_bias_with_empty_TREC(automata, in2i=in2i, i2in=i2in,)
    elif dset == 'SMS':
        mat, bias = create_mat_and_bias_with_empty_SMS(automata, in2i=in2i, i2in=i2in,)

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, config.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)

    # creating language mask for regularization
    n_vocab_extend, _ = V_embed_extend.shape
    language_mask = torch.ones(n_vocab_extend)
    language_mask[[t2i[i] for i in language]] = 0

    # for V_embed_weighted mask and extend the wildcard mat to the right dimension
    S, _ = wildcard_mat.shape
    wildcard_mat_origin_extend = np.zeros((S + config.additional_state, S + config.additional_state))
    wildcard_mat_origin_extend[:S, :S] = wildcard_mat
    wildcard_mat_origin_extend = torch.from_numpy(wildcard_mat_origin_extend).float()
    if torch.cuda.is_available():
        language_mask = language_mask.cuda()
        wildcard_mat_origin_extend = wildcard_mat_origin_extend.cuda()

    if config.normalize_automata != 'none':

        D1_avg = get_average(D1, config.normalize_automata)
        D2_avg = get_average(D2, config.normalize_automata)
        V_embed_extend_avg = get_average(V_embed_extend, config.normalize_automata)
        factor = np.float_power(D1_avg* D2_avg* V_embed_extend_avg, 1/3)
        print(factor)
        print(D1_avg)
        print(D2_avg)
        print(V_embed_extend_avg)

        D1 = D1 * (factor / D1_avg)
        D2 = D2 * (factor / D2_avg)
        V_embed_extend = V_embed_extend * (factor / V_embed_extend_avg)

    return V_embed_extend, pretrain_embed_extend, mat, bias, D1, D2, language_mask, language, wildcard_mat, wildcard_mat_origin_extend



def save_args_and_results(args, results, loggers):
    print('Saving Args and Results')
    mkdir('../model/{}'.format(args['run']))
    datetime_str = create_datetime_str()
    file_save_path = "../model/{}/{}.res".format(
        args['run'], datetime_str,
    )
    print('Saving Args and Results at: {}'.format(file_save_path))
    pickle.dump({
        'args': args,
        'res': results,
        'loggers': loggers
    }, open(file_save_path, 'wb'))


def train_onehot(args, paths):
    logger = Logger()

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

    shots = int(len(train_query) * args.train_portion)
    assert args.train_portion == 1.0
    # We currently not support ublabel and low-resource for onehot
    intent_data_train = ATISIntentBatchDataset(train_query, train_lengths, intent_train, shots)
    intent_data_dev = ATISIntentBatchDataset(dev_query, dev_lengths, intent_dev, shots)
    intent_data_test = ATISIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)


    automata_dicts = load_pkl(paths[0])
    if 'automata' not in automata_dicts:
        automata = automata_dicts
    else:
        automata = automata_dicts['automata']

    language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
    complete_tensor = language_tensor + wildcard_mat

    assert args.additional_state == 0

    if args.dataset == 'ATIS':
        mat, bias = create_mat_and_bias_with_empty_ATIS(automata, in2i=in2i, i2in=i2in,)
    elif args.dataset == 'TREC':
        mat, bias = create_mat_and_bias_with_empty_TREC(automata, in2i=in2i, i2in=i2in,)
    elif args.dataset == 'SMS':
        mat, bias = create_mat_and_bias_with_empty_SMS(automata, in2i=in2i, i2in=i2in,)

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                  config=args,
                                  mat=mat,
                                  bias=bias)

    mode = 'onehot'
    if args.loss_type == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_type == 'NormalizeNLL':
        criterion = relu_normalized_NLLLoss
    else:
        print("Wrong loss function")

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    acc_train_init, avg_loss_train_init, p, r = val(model, intent_dataloader_train, epoch=0, mode='TRAIN', config=args,
                                              i2in=i2in, criterion=criterion)
    # DEV
    acc_dev_init, avg_loss_dev_init, p, r = val(model, intent_dataloader_dev, epoch=0, mode='DEV', config=args, i2in=i2in, criterion=criterion)
    # TEST
    acc_test_init, avg_loss_test_init, p, r = val(model, intent_dataloader_test, epoch=0, mode='TEST', config=args,
                                            i2in=i2in, criterion=criterion)

    best_dev_acc = acc_dev_init
    counter = 0
    best_dev_test_acc = acc_test_init

    for epoch in range(1, args.epoch + 1):
        avg_loss = 0
        acc = 0

        pbar_train = tqdm(intent_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        model.train()
        for batch in pbar_train:

            optimizer.zero_grad()

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            scores = model(x, lengths)
            loss_cross_entropy = criterion(scores, label)
            loss = loss_cross_entropy

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            acc += (scores.argmax(1) == label).sum().item()

            pbar_train.set_postfix_str("{} - total right: {}, total loss: {}".format('TRAIN', acc, loss))

        acc = acc / len(intent_data_train)
        avg_loss = avg_loss / len(intent_data_train)
        print("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))

        # DEV
        acc_dev, avg_loss_dev, p, r = val(model, intent_dataloader_dev, epoch, 'DEV', logger, config=args, criterion=criterion)
        # TEST
        acc_test, avg_loss_test, p, r = val(model, intent_dataloader_test, epoch, 'TEST', logger, config=args, criterion=criterion)

        counter += 1  # counter for early stopping

        if (acc_dev is None) or (acc_dev > best_dev_acc):
            counter = 0
            best_dev_acc = acc_dev
            best_dev_test_acc = acc_test

        if counter > args.early_stop:
            break

    return acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, logger.record


def train_fsa_rnn(args, paths):
    logger = Logger()

    # config = Config_Integrate(args)

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
    shots = int(len(train_query) * args.train_portion)
    if args.use_unlabel:
        all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test = PredictByRE(args)
        intent_data_train = ATISIntentBatchDatasetUtilizeUnlabel(train_query, train_query_inverse, train_lengths, intent_train, all_pred_train, all_out_train, shots)
    elif args.train_portion == 0:
        # special case when train portion==0 and do not use unlabel data, should have no data
        intent_data_train = None
    else:
        intent_data_train = ATISIntentBatchDatasetBidirection(train_query, train_query_inverse, train_lengths, intent_train, shots)

    # should have no/few dev data in low-resource setting
    if args.train_portion == 0:
        intent_data_dev = None
    elif args.train_portion <= 0.01:
        intent_data_dev = ATISIntentBatchDatasetBidirection(dev_query, dev_query_inverse, dev_lengths, intent_dev, shots)
    else:
        intent_data_dev = ATISIntentBatchDatasetBidirection(dev_query, dev_query_inverse, dev_lengths, intent_dev)
    intent_data_test = ATISIntentBatchDatasetBidirection(test_query, test_query_inverse, test_lengths, intent_test, )

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz) if intent_data_train else None
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz) if intent_data_dev else None
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    print('len train dataset {}'.format(len(intent_data_train) if intent_data_train else 0))
    print('len dev dataset {}'.format(len(intent_data_dev) if intent_data_dev else 0))
    print('len test dataset {}'.format(len(intent_data_test)))
    print('num labels: {}'.format(len(in2i)))
    print('num vocabs: {}'.format(len(t2i)))

    forward_params = dict()
    forward_params['V_embed_extend'], forward_params['pretrain_embed_extend'], forward_params['mat'], forward_params['bias'], \
    forward_params['D1'], forward_params['D2'], forward_params['language_mask'], forward_params['language'], forward_params['wildcard_mat'], \
    forward_params['wildcard_mat_origin_extend'] = \
        get_init_params(args, in2i, i2in, t2i, paths[0])

    if args.bidirection:
        backward_params = dict()
        backward_params['V_embed_extend'], backward_params['pretrain_embed_extend'], backward_params['mat'], backward_params['bias'], \
        backward_params['D1'], backward_params['D2'], backward_params['language_mask'], backward_params['language'], backward_params['wildcard_mat'], \
        backward_params['wildcard_mat_origin_extend'] = \
            get_init_params(args, in2i, i2in, t2i, paths[1])


    # get h1 for FSAGRU
    h1_forward = None
    h1_backward = None
    if args.farnn == 1:
        args.farnn = 0
        temp_model = FSARNNIntegrateEmptyStateSaperateGRU(pretrained_embed=forward_params['pretrain_embed_extend'],
                                                          trans_r_1=forward_params['D1'],
                                                          trans_r_2=forward_params['D2'],
                                                          embed_r=forward_params['V_embed_extend'],
                                                          trans_wildcard=forward_params['wildcard_mat'],
                                                          config=args, )
        input_x = torch.LongTensor([[t2i['BOS']]])
        if torch.cuda.is_available():
            temp_model.cuda()
            input_x = input_x.cuda()
        h1_forward = temp_model.viterbi(input_x, None).detach()
        h1_forward = h1_forward.reshape(-1)

        if args.bidirection:
            temp_model = FSARNNIntegrateEmptyStateSaperateGRU(pretrained_embed=backward_params['pretrain_embed_extend'],
                                                              trans_r_1=backward_params['D1'],
                                                              trans_r_2=backward_params['D2'],
                                                              embed_r=backward_params['V_embed_extend'],
                                                              trans_wildcard=backward_params['wildcard_mat'],
                                                              config=args, )
            input_x = torch.LongTensor([[t2i['EOS']]])
            if torch.cuda.is_available():
                temp_model.cuda()
                input_x = input_x.cuda()
            h1_backward = temp_model.viterbi(input_x, None).detach()
            h1_backward = h1_backward.reshape(-1)

        args.farnn = 1

    if args.bidirection:
        model = IntentIntegrateSaperateBidirection_B(pretrained_embed=forward_params['pretrain_embed_extend'],
                                                     forward_params=forward_params,
                                                     backward_params=backward_params,
                                                     config=args,
                                                     h1_forward=h1_forward,
                                                     h1_backward=h1_backward)
    else:
        model = IntentIntegrateSaperate_B(pretrained_embed=forward_params['pretrain_embed_extend'],
                                          trans_r_1=forward_params['D1'],
                                          trans_r_2=forward_params['D2'],
                                          embed_r=forward_params['V_embed_extend'],
                                          trans_wildcard=forward_params['wildcard_mat'],
                                          config=args,
                                          mat=forward_params['mat'],
                                          bias=forward_params['bias'],
                                          h1_forward=h1_forward,)

    if args.loss_type == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_type == 'NormalizeNLL':
        criterion = relu_normalized_NLLLoss
    else:
        print("Wrong loss function")

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))


    # TRAIN
    acc_train_init, avg_loss_train_init, train_init_p, train_init_r = val(model, intent_dataloader_train, epoch=0, mode='TRAIN', config=args,
                                              i2in=i2in, logger=logger, criterion=criterion)
    # DEV
    acc_dev_init, avg_loss_dev_init, dev_init_p, dev_init_r = val(model, intent_dataloader_dev, epoch=0, mode='DEV', config=args, i2in=i2in, logger=logger, criterion=criterion)
    # TEST
    acc_test_init, avg_loss_test_init, test_init_p, test_init_r = val(model, intent_dataloader_test, epoch=0, mode='TEST', config=args,
                                            i2in=i2in, logger=logger, criterion=criterion)

    print("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('TRAIN', acc_train_init, avg_loss_train_init, train_init_p, train_init_r))
    print("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('DEV', acc_dev_init, avg_loss_dev_init, dev_init_p, dev_init_r))
    print("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('TEST', acc_test_init, avg_loss_test_init, test_init_p, test_init_r))
    logger.add("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('TRAIN', acc_train_init, avg_loss_train_init, train_init_p, train_init_r))
    logger.add("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('DEV', acc_dev_init, avg_loss_dev_init, dev_init_p, dev_init_r))
    logger.add("{} INITIAL: ACC: {}, LOSS: {}, P: {}, R: {}".format('TEST', acc_test_init, avg_loss_test_init, test_init_p, test_init_r))

    if args.only_probe:
        exit(0)

    best_dev_acc = acc_dev_init
    counter = 0
    best_dev_model = deepcopy(model)

    if not intent_dataloader_train: args.epoch = 0

    for epoch in range(1, args.epoch + 1):
        avg_loss = 0
        acc = 0

        pbar_train = tqdm(intent_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        model.train()
        for batch in pbar_train:

            optimizer.zero_grad()

            x_forward = batch['x_forward']
            x_backward = batch['x_backward']
            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available():
                x_forward = batch['x_forward'].cuda()
                x_backward = batch['x_backward'].cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            if args.bidirection:
                scores = model(x_forward, x_backward, lengths)
            else:
                scores = model(x_forward, lengths)

            loss = criterion(scores, label)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            acc += (scores.argmax(1) == label).sum().item()

            pbar_train.set_postfix_str("{} - total right: {}, total loss: {}".format('TRAIN', acc, loss))

        acc = acc / len(intent_data_train)
        avg_loss = avg_loss / len(intent_data_train)
        print("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))

        # DEV
        acc_dev, avg_loss_dev, p, r = val(model, intent_dataloader_dev, epoch, 'DEV', logger, config=args, criterion=criterion)
        counter += 1  # counter for early stopping

        if (acc_dev is None) or (acc_dev > best_dev_acc):
            counter = 0
            best_dev_acc = acc_dev
            best_dev_model = deepcopy(model)

        if counter > args.early_stop:
            break

    best_dev_test_acc, avg_loss_test, best_dev_test_p, best_dev_test_r \
        = val(best_dev_model, intent_dataloader_test, epoch, 'TEST', logger, config=args, criterion=criterion)

    # Save the model
    datetime_str = create_datetime_str()
    model_save_path = "../model/{}/D{:.4f}-T{:.4f}-DI{:.4f}-TI{:.4f}-{}-{}-{}".format(
        args.run, best_dev_acc, best_dev_test_acc, acc_dev_init, acc_test_init, datetime_str, args.dataset, args.seed
    )
    mkdir("../model/{}/".format(args.run))
    mkdir(model_save_path)
    print("SAVING MODEL {} .....".format(model_save_path))
    torch.save(model.state_dict(), model_save_path + '.model')


    return acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r, logger.record


def train_marry_up(args):

    assert args.additional_state == 0
    if args.model_type == 'KnowledgeDistill':
        assert args.marryup_type == 'none'
    if args.model_type == 'PR':
        assert args.marryup_type == 'none'

    all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test = PredictByRE(args)

    logger = Logger()
    # config = Config_MarryUp(args)

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

    train_query, _, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, _, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, _, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    shots = int(len(train_query) * args.train_portion)
    if args.use_unlabel:
        intent_data_train = MarryUpIntentBatchDatasetUtilizeUnlabel(train_query, train_lengths, intent_train, all_pred_train, all_out_train, shots)
    elif args.train_portion == 0:
        # special case when train portion==0 and do not use unlabel data, should have no data
        intent_data_train = None
    else:
        intent_data_train = MarryUpIntentBatchDataset(train_query, train_lengths, intent_train, all_out_train, shots)

    # should have no/few dev data in low-resource setting
    if args.train_portion == 0:
        intent_data_dev = None
    elif args.train_portion <= 0.01:
        intent_data_dev = MarryUpIntentBatchDataset(dev_query, dev_lengths, intent_dev, all_out_dev, shots)
    else:
        intent_data_dev = MarryUpIntentBatchDataset(dev_query, dev_lengths, intent_dev, all_out_dev,)
    intent_data_test = MarryUpIntentBatchDataset(test_query, test_lengths, intent_test, all_out_test)

    print('len train dataset {}'.format(len(intent_data_train) if intent_data_train else 0))
    print('len dev dataset {}'.format(len(intent_data_dev) if intent_data_dev else 0))
    print('len test dataset {}'.format(len(intent_data_test)))

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz) if intent_data_train else None
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz) if intent_data_dev else None
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    pretrained_embed = load_glove_embed('../data/{}/'.format(args.dataset), args.embed_dim)
    if args.random_embed: pretrained_embed = np.random.random(pretrained_embed.shape)

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)

    model = IntentMarryUp(
        pretrained_embed=pretrain_embed_extend,
        config=args,
        label_size=len(in2i),
    )

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))
    acc_dev_init, avg_loss_dev_init, p, r = val_marry(model, intent_dataloader_dev, 0, 'DEV', args, logger)
    # TEST
    acc_test_init, avg_loss_test_init, p, r = val_marry(model, intent_dataloader_test, 0, 'TEST', args, logger)

    best_dev_acc = acc_dev_init
    counter = 0
    best_dev_model = deepcopy(model)
    # when no training data, just run a test.
    if not intent_dataloader_train: args.epoch = 0

    for epoch in range(1, args.epoch + 1):
        avg_loss = 0
        acc = 0

        pbar_train = tqdm(intent_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        model.train()
        for batch in pbar_train:

            optimizer.zero_grad()

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']
            re_tag = batch['re']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re_tag = re_tag.cuda()

            scores = model(x, lengths, re_tag)

            loss_cross_entropy = criterion(scores, label)

            if args.model_type == 'MarryUp':
                loss = loss_cross_entropy

            elif args.model_type == 'KnowledgeDistill':
                softmax_scores = torch.log_softmax(scores, 1)
                softmax_re_tag_teacher = torch.softmax(re_tag, 1)
                loss_KL = torch.nn.KLDivLoss()(softmax_scores, softmax_re_tag_teacher)
                loss = loss_cross_entropy * args.l1 + loss_KL * (
                            1 - args.l1)  # in KD, l1 stands for the alpha controlling to learn from true / imitate teacher

            elif args.model_type == 'PR':
                log_softmax_scores = torch.log_softmax(scores, 1)
                softmax_scores = torch.softmax(scores, 1)
                product_term = torch.exp(re_tag - 1) * args.l2  #in PR, l2 stands for the regularization term, higher l2, harder rule constraint
                teacher_score = torch.mul(softmax_scores, product_term)
                softmax_teacher = torch.softmax(teacher_score, 1)
                loss_KL = torch.nn.KLDivLoss()(log_softmax_scores, softmax_teacher)
                loss = loss_cross_entropy * args.l1 + loss_KL * (
                            1 - args.l1)  # in PR, l1 stands for the alpha controlling to learn from true / imitate teacher

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            acc += (scores.argmax(1) == label).sum().item()

            pbar_train.set_postfix_str("{} - total right: {}, total loss: {}".format('TRAIN', acc, loss))

        acc = acc / len(intent_data_train)
        avg_loss = avg_loss / len(intent_data_train)
        # print("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))

        # DEV
        acc_dev, avg_loss_dev, p, r = val_marry(model, intent_dataloader_dev, epoch, 'DEV', args, logger)

        counter += 1  # counter for early stopping

        if (acc_dev is None) or (acc_dev > best_dev_acc):
            counter = 0
            best_dev_acc = acc_dev
            best_dev_model = deepcopy(model)

        if counter > args.early_stop:
            break

    best_dev_test_acc, avg_loss_test, best_dev_test_p, best_dev_test_r \
        = val_marry(best_dev_model, intent_dataloader_dev, epoch, 'TEST', args, logger)

    return acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r, logger.record