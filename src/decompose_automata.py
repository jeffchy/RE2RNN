import os
import pickle
from rules.fsa_to_tensor import dfa_to_tensor
import argparse
from data import load_classification_dataset, load_pkl, decompose_tensor_split


def decompose_automata(args):
    merged_automata = load_pkl('../data/{}/automata/{}'.format(args.dataset_name, args.automata_name))

    print('AUTOMATA TO TENSOR')
    print('Total States: {}'.format(len(merged_automata['states'])))
    # first load vocabs
    dataset = load_classification_dataset(args.dataset_name)
    word2idx = dataset['t2i']
    language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(merged_automata, word2idx)
    complete_tensor = language_tensor + wildcard_mat

    print('DECOMPOSE SPLIT AUTOMATA')

    for random_state in range(4):
        print('DECOMPOSING RANK: {}, TENSOR SIZE: {}'.format(args.rank, language_tensor.shape))
        V_embed_split, D1_split, D2_split, rec_error = \
            decompose_tensor_split(language_tensor, language, word2idx, args.rank,
                                   random_state=random_state, n_iter_max=30, init=args.init)

        save_dict = {
            'automata': merged_automata,
            'V':V_embed_split,
            'D1': D1_split,
            'D2': D2_split,
            'language': language,
            'wildcard_mat': wildcard_mat,
        }
        pickle.dump(save_dict, open('../data/{}/automata/automata.{}.{:.4f}.{}.pkl'.format(args.dataset_name, args.rank, rec_error[-1], random_state), 'wb'))

    print('FINISHED')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ATIS', help="dataset name, ATIS, TREC, SMS")
    parser.add_argument('--automata_name', type=str, default='all.1030104910-1604054950.712768.split.pkl', help="automata name prefix")
    parser.add_argument('--rank', type=int, default=150, help="rank")
    parser.add_argument('--init', type=str, default='svd', help="initialization")


    args = parser.parse_args()
    assert args.dataset_name in ['ATIS', 'TREC', 'SMS']
    assert args.init in ['svd', 'random']

    decompose_automata(args)