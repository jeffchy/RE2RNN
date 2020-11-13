import numpy as np
from os import popen
from typing import Dict


def is_number(token):
    return token.replace('.', '', 1).isdigit()


punctuations = {
    ',', '，', ':', '：', '!', '！', '《', '》', '。', '；', '.', '(', ')', '（', '）',
    '|', '?', '"'
}


def is_punct(token):
    return token in punctuations


def subtype_disturb_func(val):
    dist = np.random.normal
    disturb = abs(dist(0, 0.01))
    return val - disturb


def subtype_disturb_func_1(val):
    dist = np.random.normal
    disturb = dist(0, 0.01)
    return val - disturb


def dfa_to_tensor(automata, word2idx: Dict[str, int], subtype=False, disturb_func=subtype_disturb_func):
    """
    Parameters
    ----------
    automata: Automata.to_dict()
    word2idx

    Returns
    -------
    tensor: tensor for language
    state2idx: state to idx
    wildcard_mat: matrix for wildcard
    language: set for language
    """

    all_states = list(automata['states'])
    state2idx = {
        state: idx for idx, state in enumerate(all_states)
    }

    number_idxs = {word: idx for word, idx in word2idx.items() if is_number(word)}
    punct_idxs = {word: idx for word, idx in word2idx.items() if is_punct(word)}

    max_states = len(automata['states'])
    tensor = np.zeros((len(word2idx), max_states, max_states))

    language = set([])
    language.update(number_idxs.keys())
    language.update(punct_idxs.keys())

    wildcard_mat = np.zeros((max_states, max_states))

    for fr_state, to in sorted(automata['transitions'].items()):
        for to_state, to_edges in sorted(to.items()):
            for edge in to_edges:

                val = 1
                if (subtype) and ((fr_state in automata['subtypes']) or (to_state in automata['subtypes'])): # add subtype distortion
                    val = disturb_func(1)

                if edge == '&': # punctuations
                    tensor[list(punct_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                elif edge == '%': # digits
                    tensor[list(number_idxs.values()), state2idx[fr_state], state2idx[to_state]] = val
                elif edge == "$":
                    wildcard_mat[state2idx[fr_state], state2idx[to_state]] = val
                else:
                    if edge in word2idx:
                        tensor[word2idx[edge], state2idx[fr_state], state2idx[to_state]] = val
                        language.add(edge)
                    else:
                        # print('OOV word: {} in rule'.format(edge))
                        pass
    return tensor, state2idx, wildcard_mat, list(language)


class Automata:
    """class to represent an Automata"""

    def __init__(self, language=set(['0', '1'])):
        self.states = set()
        self.startstate = None
        self.finalstates = []
        self.finalstates_label = {}
        self.transitions = dict()
        self.language = language

    def to_dict(self):
        return {
            'states': self.states,
            'startstate': self.startstate,
            'finalstates': self.finalstates,
            'transitions': self.transitions,
            'language': self.language,
            'finalstates_label': self.finalstates_label
        }

    def setstartstate(self, state):
        self.startstate = state
        self.states.add(state)

    def addfinalstates(self, state):
        if isinstance(state, int):
            state = [state]
        for s in state:
            if s not in self.finalstates:
                self.finalstates.append(s)

    def addfinalstates_label(self, state, label):
        if isinstance(state, int):
            state = [state]
        assert label not in self.finalstates_label
        self.finalstates_label[label] = state

    def addtransition(self, fromstate, tostate, inp):
        if isinstance(inp, str):
            inp = set([inp])
        self.states.add(fromstate)
        self.states.add(tostate)
        if fromstate in self.transitions:
            if tostate in self.transitions[fromstate]:
                self.transitions[fromstate][tostate] = self.transitions[fromstate][tostate].union(inp)
            else:
                self.transitions[fromstate][tostate] = inp
        else:
            self.transitions[fromstate] = {tostate: inp}

    def addtransition_dict(self, transitions):
        for fromstate, tostates in transitions.items():
            for state in tostates:
                self.addtransition(fromstate, state, tostates[state])

    def getDotFile(self):
        dotFile = "digraph DFA {\nrankdir=LR\n"
        if len(self.states) != 0:
            dotFile += "root=s1\nstart [shape=point]\nstart->s%d\n" % self.startstate
            for state in self.states:
                if state in self.finalstates:
                    dotFile += "s%d [shape=doublecircle]\n" % state
                else:
                    dotFile += "s%d [shape=circle]\n" % state
            for fromstate, tostates in self.transitions.items():
                for state in tostates:
                    for char in tostates[state]:
                        dotFile += 's%d->s%d [label="%s"]\n' % (fromstate, state, char)
        dotFile += "}"
        return dotFile

    def drawGraph(self, file="",):
        """From https://github.com/max99x/automata-editor/blob/master/util.py"""
        f = popen(r"dot -Tpng -o %s.png" % file, 'w')
        try:
            f.write(self.getDotFile())
        except:
            raise BaseException("Error creating graph")
        finally:
            f.close()


def drawGraph(automata, file="",):
    """From https://github.com/max99x/automata-editor/blob/master/util.py"""
    f = popen(r"dot -Tpng -o %s.png" % file, 'w')
    try:
        f.write(automata.getDotFile())
    except:
        raise BaseException("Error creating graph")
    finally:
        f.close()
