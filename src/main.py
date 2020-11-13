import sys
sys.path.append('../../')
sys.path.append('../')
import argparse

from src.train import train_fsa_rnn, train_marry_up, train_onehot, save_args_and_results
from src.utils.utils import set_seed, get_automata_from_seed



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument('--max_state', type=int, default=3, help="max state of each FSARNN")
    parser.add_argument('--bz', type=int, default=500, help="batch size")
    parser.add_argument('--epoch', type=int, default=200, help="max state of each FSARNN")
    parser.add_argument('--seq_max_len', type=int, default=30, help="max length of sequence")
    parser.add_argument('--early_stop', type=int, default=20, help="number of epochs that apply early stopping if dev metric not improve")
    parser.add_argument('--run', type=str, default="integrate", help="run folder name to save model")
    parser.add_argument('--seed', type=str, default='0:1:2:3', help="random seed")
    parser.add_argument('--model_id', type=str, default='none', help="model id, identifying model")
    parser.add_argument('--activation', type=str, default='relu', help='nonlinear of model, [tanh, sigmoid, relu, relu6]')
    parser.add_argument('--clamp_hidden', type=int, default=0, help='if constrain the hidden layer at each time stamp to between 0, 1')

    parser.add_argument('--train_fsa', type=int, default=1, help="if we train the fsa parameters")
    parser.add_argument('--train_wildcard', type=int, default=0, help='If we train the wildcard matrix')
    parser.add_argument('--train_linear', type=int, default=0, help="if we train the linear parameters")
    parser.add_argument('--train_V_embed', type=int, default=0, help="if we train the V_embed parameters")
    parser.add_argument('--train_word_embed', type=int, default=0, help="if we train the word embed parameters")
    parser.add_argument('--train_beta', type=int, default=0, help="if we train the vector beta")
    parser.add_argument('--clip_neg', type=int, default=0, help="if we use relu after each hidden update")
    parser.add_argument('--sigmoid_exponent', type=int, default=5, help="exponent in TENSORFSAGRU sigmoid")
    parser.add_argument('--beta', type=float, default=1.0, help="in [0,1] interval, 1 means use regex, 0 means only use embedding")
    parser.add_argument('--farnn', type=int, default=0, help="0 if use farnn, 1 if use fagru")
    parser.add_argument('--bias_init', type=float, default=7, help="bias_init for gru")
    parser.add_argument('--reg_type', type=str, default='nuc', help="nuclear norm or fro norm")
    parser.add_argument('--wfa_type', type=str, default='viterbi', help='forward or viterbi')
    parser.add_argument('--random_embed', type=int, default=0, help='random embedding or not')
    parser.add_argument('--clamp_score', type=int, default=0, help='if we clamp the score in [0, 1]')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='rnn hidden dim')
    parser.add_argument('--model_type', type=str, default='FSARNN', help='baseline MarryUp or FSARNN')
    parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer SGD or ADAM')
    parser.add_argument('--additional_nonlinear', type=str, default='none', help='additional nonlinear after DxR, should be in [none, relu, sigmoid, tanh]')
    parser.add_argument('--additional_state', type=int, default=0, help='additional reserved state for generalization')
    parser.add_argument('--dataset', type=str, default='ATIS', help="dataset name, in ATIS, TREC, SMS")
    parser.add_argument('--regularization', type=str, default='V_embed_weighted', help="[V_embed, V_embed_weighted, D1, D2, None]")
    parser.add_argument('--only_probe', type=int, default=0, help='if we only prob and not train')
    parser.add_argument('--rnn', type=str, default='RNN', help='rnn type only for MarryUp Baseline')
    parser.add_argument('--re_tag_dim', type=int, default=20, help='re tag dim only for MarryUp Baseline')
    parser.add_argument('--marryup_type', type=str, default='input', help='only for marry up baseline, should be in [input, output, all, none]')
    parser.add_argument('--train_portion', type=float, default=1.0, help='training portion, in 0.01, 0.1, 1.0 ')
    parser.add_argument('--random', type=int, default=0, help='if use random initialzation')

    parser.add_argument('--l1', type=float, default=0, help="kd alpha")
    parser.add_argument('--l2', type=float, default=0., help="pr constant")

    parser.add_argument('--bidirection', type=int, default=0, help='if the model is bidirection')
    parser.add_argument('--xavier', type=int, default=1, help='if the FSAGRU model initialized using xavier')
    parser.add_argument('--normalize_automata', type=str, default='none', help='how to normalize the automata, none, l1, l2, avg')
    parser.add_argument('--random_noise', type=float, default=0.001, help='random noise used when adding additional states')
    parser.add_argument('--loss_type', type=str, default='CrossEntropy', help='CrossEntropy, NormalizeNLL')
    parser.add_argument('--use_unlabel', type=int, default=0, help='use unlabel or not')
    parser.add_argument('--embed_dim', type=int, default=100, help='embed dim')

    parser.add_argument('--automata_path_forward', type=str, default='none', help="automata path")
    parser.add_argument('--automata_path_backward', type=str, default='none', help="automata path")

    args = parser.parse_args()
    assert args.farnn in [0, 1]

    seeds = [int(i) for i in args.seed.split(':')]
    results = {}
    loggers = {}
    for seed in seeds:
        assert seed in [0, 1, 2, 3]
        set_seed(seed)
        if args.model_type == 'FSARNN':
            automata_path_forward, automata_path_backward = get_automata_from_seed(args, seed)
            paths = (automata_path_forward, automata_path_backward)
            args.automata_path_forward = automata_path_forward
            args.automata_path_backward = automata_path_backward
            acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r, logger_res = train_fsa_rnn(args, paths)
            results[seed] = [acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r,]
            print("Results: ", results[seed])
            loggers[seed] = logger_res
        elif args.model_type in ['MarryUp', 'KnowledgeDistill', 'PR']:
            assert args.rnn in ['RNN', 'LSTM', 'GRU', 'DAN', 'CNN']
            assert args.marryup_type in ['input', 'output', 'all', 'none']
            if args.rnn in ['DAN', 'CNN']:
                assert args.bidirection == 0
            automata_path_forward, automata_path_backward = get_automata_from_seed(args, seed)
            paths = (automata_path_forward, automata_path_backward)
            args.automata_path_forward = automata_path_forward
            args.automata_path_backward = automata_path_backward
            acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r,logger_res = train_marry_up(args)
            results[seed] = [acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc,best_dev_test_p, best_dev_test_r, ]
            loggers[seed] = logger_res
        elif args.model_type == 'Onehot':
            automata_path_forward, automata_path_backward = get_automata_from_seed(args, seed)
            paths = (automata_path_forward, automata_path_backward)
            args.automata_path_forward = automata_path_forward
            args.automata_path_backward = automata_path_backward
            acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, logger_res = train_onehot(args, paths)
            results[seed] = [acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc ]
            loggers[seed] = logger_res

    args_value_dict = args.__dict__
    save_args_and_results(args_value_dict, results, loggers)



