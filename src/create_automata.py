import os
import pickle
from rules.load_data_and_rules import load_TREC_dataset, load_SMS_dataset, load_rule
from pydash.arrays import compact
from automata_tools import NFAtoDFA, DFAtoMinimizedDFA
from rules.dfa_from_rule import NFAFromRegex
from rules.fsa_to_tensor import Automata, drawGraph
from src.utils.utils import mkdir, create_datetime_str
import argparse

def create_dataset_automata(args):
    if args.dataset_name == 'TREC':
        rules = load_TREC_dataset()['rules']
    elif args.dataset_name == 'SMS':
        rules = load_SMS_dataset()['rules']
    elif args.dataset_name == 'ATIS':
        rules = load_rule(os.path.join('../data/ATIS', 'rules.v0.config'))
    else:
        raise ValueError('WRONG DATASET NAME')

    reversed = bool(args.reversed)

    mode = 'reversed' if reversed else 'split'

    print('GETTING RULE FILES')

    automaton = {}
    all_automata = Automata()
    all_automata.setstartstate(0)
    state_idx = 1

    print('COLLECTING ALL AUTOMATAS')
    mkdir('../data/{}/automata'.format(args.dataset_name))
    for indexClass, rulesOfAClass in rules.iterrows():

        concatenatedRule = f'(({")|(".join(compact(rulesOfAClass))}))'
        print('\n')
        print(concatenatedRule)
        nfa = NFAFromRegex().buildNFA(concatenatedRule, reversed=reversed)
        dfa = NFAtoDFA(nfa)
        minDFA = DFAtoMinimizedDFA(dfa)
        automaton[indexClass] = minDFA
        drawGraph(minDFA, '../data/{}/automata/{}.{}'.format(args.dataset_name, indexClass, mode))

    print('MERGING AUTOMATA')
    for label, automata in automaton.items():
        tok = 'BOS'
        if reversed:
            tok = 'EOS'
        all_automata.addtransition(0, state_idx, tok)  # may cause bug when the RE is accosiate with BOS
        states = list(automata.states)
        final_states = list(automata.finalstates)
        num_states = len(states)
        used_states = [i for i in range(state_idx, state_idx + num_states)]
        states2idx = {states[i]: used_states[i] for i in range(num_states)}

        for fr_state, to in automata.transitions.items():
            for to_state, to_edges in to.items():
                for edge in to_edges:
                    all_automata.addtransition(states2idx[fr_state], states2idx[to_state], edge)

        all_automata.addfinalstates([states2idx[i] for i in final_states])
        all_automata.addfinalstates_label([states2idx[i] for i in final_states], label)
        state_idx += (num_states)

    merged_automata = all_automata.to_dict()
    time_str = create_datetime_str()

    path = '../data/{}/automata/{}.{}.{}'.format(args.dataset_name, args.automata_name, time_str, mode)
    print("Drawing Graph and save at: {}".format(path))
    drawGraph(all_automata, )
    path = '../data/{}/automata/{}.{}.{}.pkl'.format(args.dataset_name, args.automata_name, time_str, mode)
    print("Save the automata object at: {}".format(path))
    pickle.dump(merged_automata, open(path, 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ATIS', help="dataset name, ATIS, TREC, SMS")
    parser.add_argument('--automata_name', type=str, default='all', help="automata name prefix")
    parser.add_argument('--reversed', type=int, default=0, help="if we reverse the string")

    args = parser.parse_args()
    assert args.dataset_name in ['ATIS', 'TREC', 'SMS']

    create_dataset_automata(args)