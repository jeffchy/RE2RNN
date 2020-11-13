import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(_project_root)

import pandas as pd
import re
from typing import List, Optional
from pydash.arrays import compact
from automata_tools import get_word_to_index
from rules.dfa_from_rule import tokenizer

MITR_PATH = os.path.join(_project_root, 'data', 'MITR')
SMS_PATH = os.path.join(_project_root, 'data', 'SMS')
TREC_PATH = os.path.join(_project_root, 'data', 'TREC')
Youtube_PATH = os.path.join(_project_root, 'data', 'Youtube')


def load_rule(filePath: str):
    """
    Load rule in pd.DataFrame that use Tag name as index, each column contains a rule or None

    ### Example
    location                          None                                  None
    Cuisine                           None                                  None
    Price                             None                     (([0-9]*)|vmost)* 
    Rating                            None                     (open|closew* ){0
    Hours            ( night| dinner| l...                                  None
    Amenity                           None                                  None
    Restaurant_Name                   None                                  None
    """
    ruleOfTags = dict()
    with open(filePath, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        currentTag: Optional[str] = None
        currentRules: List[str] = []
        for line in lines:
            # if this line is a tag name like "[Cuisine]"
            if re.match(r'^\[[a-z_A-Z_\+]+\]$', line):
                tagName = line[1:-1]
                # if we are going to load a new set of rules of a tag
                if tagName != currentTag:
                    # if we were parsing previous set of rules, store them into dict
                    if currentTag != None:

                        # if currentTag in ['ABBREVIATION', 'HUMAN']:
                        #     ruleOfTags[currentTag] = compact(currentRules)
                        ruleOfTags[currentTag] = compact(currentRules)

                        currentRules = []
                    currentTag = tagName
            else:  # parsing line that contains a rule and a example split by //
                rule = line.split('//')[0]
                currentRules.append(rule.strip())
    # add rules of last tag
    if currentTag != None:
        # if currentTag in ['ABBREVIATION', 'HUMAN']:
        #     ruleOfTags[currentTag] = compact(currentRules)
        ruleOfTags[currentTag] = compact(currentRules)

    return pd.DataFrame.from_dict(ruleOfTags, orient='index')

def load_SMS_dataset():
    testDf = pd.read_csv(os.path.join(SMS_PATH, 'test.csv')).reset_index(drop=True).rename({'v1': 'class', 'v2': 'text'}, axis='columns')
    testDf['mode'] = 'test'
    testDf['text'] = testDf['text'].str.lower()
    trainDf = pd.read_csv(os.path.join(SMS_PATH, 'train.csv')).reset_index(drop=True).rename({'v1': 'class', 'v2': 'text'}, axis='columns')
    trainDf['mode'] = 'train'
    trainDf['text'] = trainDf['text'].str.lower()
    validDf = pd.read_csv(os.path.join(SMS_PATH, 'valid.csv')).reset_index(drop=True).rename({'v1': 'class', 'v2': 'text'}, axis='columns')
    validDf['mode'] = 'valid'
    validDf['text'] = validDf['text'].str.lower()
    df = pd.concat([testDf, trainDf, validDf], ignore_index=True)
    indexToWord, wordToIndex = get_word_to_index(map(lambda text: tokenizer(text) + text.split(' '), list(df['text'])))
    return {
        'rules': load_rule(os.path.join(SMS_PATH, 'rules.config')),
        'data': df,
        'indexToWord': indexToWord,
        'wordToIndex': wordToIndex
    }

def load_TREC_dataset():
    rawRows: List[str] = []
    texts: List[str] = []
    classes: List[str] = []
    modeTag: List[str] = [] # train, valid, test
    with open(os.path.join(TREC_PATH, 'train.txt'), 'r', encoding='utf8') as f:
        dataRows = list(map(lambda line: line.strip(), f.read().split('\n')))
        rawRows += dataRows
        modeTag += map(lambda _: 'train', dataRows)
    with open(os.path.join(TREC_PATH, 'valid.txt'), 'r', encoding='utf8') as f:
        dataRows = list(map(lambda line: line.strip(), f.read().split('\n')))
        rawRows += dataRows
        modeTag += map(lambda _: 'valid', dataRows)
    with open(os.path.join(TREC_PATH, 'test.txt'), 'r', encoding='utf8') as f:
        dataRows = list(map(lambda line: line.strip(), f.read().split('\n')))
        rawRows += dataRows
        modeTag += map(lambda _: 'test', dataRows)
    for rowText in rawRows:
        result = re.match(r'^(?P<class>[A-Z]+):(?P<text>.+)$', rowText)
        # if result['class'] not in ['ABBREVIATION', 'HUMAN']:
        #     classes.append(result['class'])
        #     texts.append(result['text'].lower())
        classes.append(result['class'])
        texts.append(result['text'].lower())

    df = pd.DataFrame(zip(classes, texts, modeTag)).rename(columns={0: 'class', 1: 'text', 2: 'mode'})
    textsToIndex = list(map(lambda text: tokenizer(text) + text.split(' '), list(df['text'])))
    indexToWord, wordToIndex = get_word_to_index(textsToIndex)
    return {
        'rules': load_rule(os.path.join(TREC_PATH, 'rules.old.config')),
        'data': df,
        'indexToWord': indexToWord,
        'wordToIndex': wordToIndex
    }


if __name__ == "__main__":
    # print(load_TREC_dataset()['data'].groupby('mode').count())
    print(load_TREC_dataset()['data'])
    # load_SMS_dataset()
