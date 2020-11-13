import numpy as np

def create_mat_and_bias_with_empty_ATIS(automata, in2i, i2in,):
    # create
    mat = np.zeros((len(automata['states']) , len(i2in)))
    bias = np.zeros((len(i2in),))

    # extract final states, for multiple final states, use OR gate
    for lab, states in automata['finalstates_label'].items():
        lab_idx = in2i[lab]
        for state in states:
            mat[state, lab_idx] = 1

    # for '+', use AND gate
    # for label_idx, label in i2in.items():
    #     if '+' in label:
    #         asso_labels = label.split('+')
    #         bias[label_idx] += (- len(asso_labels) + 1)
    #         for label in asso_labels:
    #             asso_states = automata['finalstates_label'][label]
    #             for state in asso_states:
    #                 mat[state, label_idx] = 1

    # Priority
    mat[automata['finalstates_label']['ground_fare'], in2i['ground_service']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['city']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['airport']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['airline']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['meal']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['capacity']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['cheapest']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['distance']] -= 1.0
    mat[automata['finalstates_label']['airfare'], in2i['flight']] -= 1.0
    mat[automata['finalstates_label']['flight'], in2i['flight+airline']] -= 1.0
    mat[automata['finalstates_label']['abbreviation'], in2i['restriction']] -= 1.0
    mat[automata['finalstates_label']['ground_service'], in2i['airport']] -= 1.0

    return mat, bias

def create_mat_and_bias_with_empty_MITR(automata, s2i, i2s):
    # create
    mat = np.zeros((len(automata['states']) , len(i2s)))
    bias = np.zeros((len(i2s),))

    # extract final states, for multiple final states, use OR gate
    for lab, states in automata['finalstates_label'].items():
        lab_idx = s2i[lab]
        for state in states:
            mat[state, lab_idx] = 1

    # for '+', use AND gate
    for label_idx, label in i2s.items():
        if '+' in label:
            asso_labels = label.split('+')
            bias[label_idx] += (- len(asso_labels) + 1)
            for label in asso_labels:
                asso_states = automata['finalstates_label'][label]
                for state in asso_states:
                    mat[state, label_idx] = 1

    # Priority

    return mat, bias

def create_mat_and_bias_with_empty_TREC(automata, in2i, i2in):
    # create
    mat = np.zeros((len(automata['states']) , len(i2in)))
    bias = np.zeros((len(i2in),))

    # extract final states, for multiple final states, use OR gate
    for lab, states in automata['finalstates_label'].items():
        lab_idx = in2i[lab]
        for state in states:
            mat[state, lab_idx] = 1

    # Priority
    # mat[automata['finalstates_label']['HUMAN'], in2i['ABBREVIATION']] -= 1.0
    mat[automata['finalstates_label']['ENTITY'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['HUMAN'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['NUMERIC'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['LOCATION'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['ENTITY'], in2i['ABBREVIATION']] -= 1.0
    # mat[automata['finalstates_label']['DESCRIPTION'], in2i['LOCATION']] -= 1.0

    # mat[automata['finalstates_label']['ENTITY'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['ABBREVIATION'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['LOCATION'], in2i['DESCRIPTION']] -= 1.0
    # mat[automata['finalstates_label']['HUMAN'], in2i['ENTITY']] -= 1.0

    return mat, bias

def create_mat_and_bias_with_empty_SMS(automata, in2i, i2in):
    # create
    mat = np.zeros((len(automata['states']) , len(i2in)))
    bias = np.zeros((len(i2in),))

    # extract final states, for multiple final states, use OR gate
    for lab, states in automata['finalstates_label'].items():
        lab_idx = in2i[lab]
        for state in states:
            mat[state, lab_idx] = 1

    # Priority
    mat[automata['finalstates_label']['spam'], in2i['ham']] -= 1.0

    return mat, bias