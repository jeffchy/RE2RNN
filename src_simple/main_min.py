import sys
sys.path.append('../../')
sys.path.append('../')
import argparse

from src_simple.train_farnn import train_fsa_rnn, train_onehot
from src.utils.utils import set_seed, get_automata_from_seed
from copy import deepcopy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument('--max_state', type=int, default=3, help="max state of each FSARNN")
    parser.add_argument('--bz', type=int, default=500, help="batch size")
    parser.add_argument('--epoch', type=int, default=30, help="max state of each FSARNN")
    parser.add_argument('--seq_max_len', type=int, default=30, help="max length of sequence")
    parser.add_argument('--early_stop', type=int, default=20, help="number of epochs that apply early stopping if dev metric not improve")
    parser.add_argument('--run', type=str, default="integrate", help="run folder name to save model")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--activation', type=str, default='none', help='nonlinear of model, [tanh, relu, none]')

    parser.add_argument('--train_fsa', type=int, default=1, help="if we train the fsa parameters")
    parser.add_argument('--train_wildcard', type=int, default=0, help='If we train the wildcard matrix')
    parser.add_argument('--train_linear', type=int, default=0, help="if we train the linear parameters")
    parser.add_argument('--train_V_embed', type=int, default=0, help="if we train the V_embed parameters")
    parser.add_argument('--train_word_embed', type=int, default=0, help="if we train the word embed parameters")
    parser.add_argument('--train_beta', type=int, default=0, help="if we train the vector beta")
    parser.add_argument('--beta', type=float, default=1.0, help="in [0,1] interval, 1 means use regex, 0 means only use embedding")
    parser.add_argument('--farnn', type=int, default=0, help="0 if use farnn, 1 if use fagru")
    parser.add_argument('--bias_init', type=float, default=7, help="bias_init for gru")
    parser.add_argument('--wfa_type', type=str, default='viterbi', help='forward or viterbi')
    parser.add_argument('--clamp_score', type=int, default=0, help='if we clamp the score in [0, 1]')
    parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer SGD or ADAM')
    parser.add_argument('--additional_nonlinear', type=str, default='none', help='additional nonlinear after DxR, should be in [none, relu, sigmoid, tanh]')
    parser.add_argument('--additional_state', type=int, default=0, help='additional reserved state for generalization')
    parser.add_argument('--dataset', type=str, default='ATIS', help="dataset name, in ATIS, TREC, SMS")
    parser.add_argument('--train_portion', type=float, default=1.0, help='training portion, in 0.01, 0.1, 1.0 ')
    parser.add_argument('--random', type=int, default=0, help='if use random initialzation')

    parser.add_argument('--bidirection', type=int, default=0, help='if the model is bidirection')
    parser.add_argument('--normalize_automata', type=str, default='l2', help='how to normalize the automata, none, l1, l2, avg')
    parser.add_argument('--random_noise', type=float, default=0.001, help='random noise used when adding additional states')
    parser.add_argument('--embed_dim', type=int, default=100, help='embed dim')

    parser.add_argument('--automata_path_forward', type=str, default='none', help="automata path")
    parser.add_argument('--automata_path_backward', type=str, default='none', help="automata path")

    parser.add_argument('--model_type', type=str, default='FSARNN', help='baseline MarryUp or FSARNN')

    args = parser.parse_args()
    args_bak = deepcopy(args)

    assert args.farnn in [0, 1]

    results = {}
    loggers = {}
    seed = args.seed
    set_seed(args.seed)
    if args.model_type == 'FSARNN':
        automata_path_forward, automata_path_backward = get_automata_from_seed(args_bak, seed)
        paths = (automata_path_forward, automata_path_backward)
        args.automata_path_forward = automata_path_forward
        args.automata_path_backward = automata_path_backward
        train_fsa_rnn(args, paths)

    elif args.model_type == 'Onehot':
        automata_path_forward, automata_path_backward = get_automata_from_seed(args_bak, seed)
        paths = (automata_path_forward, automata_path_backward)
        args.automata_path_forward = automata_path_forward
        args.automata_path_backward = automata_path_backward
        train_onehot(args, paths)








