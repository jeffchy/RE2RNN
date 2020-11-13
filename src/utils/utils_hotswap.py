from src.data import load_classification_dataset, create_vocabs
import pickle

def modify_TREC_dataset():
    dataset = load_classification_dataset('TREC')
    expire_intents = ['ABBREVIATION', 'HUMAN']
    expire_intents_indice = [dataset['in2i'][i] for i in expire_intents]

    new_query_train = []
    new_query_dev = []
    new_query_test= []
    new_intent_train = []
    new_intent_dev = []
    new_intent_test = []

    for dset in ['train', 'dev', 'test']:
        for i in range(len(dataset['intent_{}'.format(dset)])):
            if dataset['intent_{}'.format(dset)][i] not in expire_intents_indice:
                if dset == 'train':
                    new_intent_train.append(dataset['intent_{}'.format(dset)][i])
                    new_query_train.append(dataset['query_{}'.format(dset)][i])
                elif dset == 'dev':
                    new_intent_dev.append(dataset['intent_{}'.format(dset)][i])
                    new_query_dev.append(dataset['query_{}'.format(dset)][i])
                else:
                    new_intent_test.append(dataset['intent_{}'.format(dset)][i])
                    new_query_test.append(dataset['query_{}'.format(dset)][i])

    all_ins = [i for i in dataset['in2i'].keys() if i not in expire_intents]
    new_i2in, new_in2i  = create_vocabs(all_ins, 'labels')

    for i in range(len(new_intent_train)):
        new_intent_train[i] = new_in2i[dataset['i2in'][new_intent_train[i]]]

    for i in range(len(new_intent_dev)):
        new_intent_dev[i] = new_in2i[dataset['i2in'][new_intent_dev[i]]]

    for i in range(len(new_intent_test)):
        new_intent_test[i] = new_in2i[dataset['i2in'][new_intent_test[i]]]

    print('SAVING NEW DATASET')
    new_dataset = {
        't2i': dataset['t2i'], 'i2t': dataset['i2t'], 'in2i': new_in2i, 'i2in': new_i2in,
        'query_train': new_query_train, 'intent_train': new_intent_train,
        'query_dev': new_query_dev, 'intent_dev': new_intent_dev,
        'query_test': new_query_test, 'intent_test': new_intent_test,
    }
    pickle.dump(new_dataset, open('../data/{}/dataset.hotswap.pkl'.format('TREC'), 'wb'))


if __name__ == '__main__':
    modify_TREC_dataset()




