from src.rules.rule_tokenizer import ruleParser
from typing import List


def reverse_regex(tokenized_re: List[str]):
    tokenized_re.reverse()
    operator = {'*', '+', '?'}
    reversed_re = []
    stack = []
    prev = None
    temp_stack = []
    set_begin = False
    for i in tokenized_re:
        if i == '}':
            set_begin = True

        if set_begin:
            temp_stack.append(i)
            if i == '{':
                set_begin = False
            prev = i
            continue

        if prev == '{':
            temp_stack.append(i)
            temp_stack.reverse()
            reversed_re += temp_stack
            prev = i
            temp_stack = []
            continue

        if i == '(':
            # pop from stack
            reversed_re.append(')')
            pop_tok = stack.pop(-1)
            if pop_tok != '':
                reversed_re.append(pop_tok)

        elif i == ')':
            # add to stack
            if prev in operator:
                stack.append(prev)
            else:
                stack.append('')

            reversed_re.append('(')

        elif i in operator:
            pass

        else:
            reversed_re.append(i)
            if prev in operator:
                reversed_re.append(prev)

        prev = i

    return reversed_re


if __name__ == '__main__':

    # ruleString = "(($* name $? (a|one|two|three|four|five|six|seven|eight|nine|ten) $+)|(what $* ( played | play | run | study | studied | patent ) $*)|($ * ( which | what ) $* ( team | group | groups | teams ) $*)|($* ( which | (what (is|was|were|are | 's)|what's) ) $? (article|articles|animal|animals|book|books|card|cards|drug|drugs|drink|drinks|film|films|food|foods|game|games|instrument|instruments|language|languages|movie|movies|novel|novels|play|plays|product|products|plant|plants|ship|ships|system|systems|symbol|symbols|trial|trials|train|trains|weapon|weapons|sword|swords|word|words) $*)|($* ( which | what ) $? (article|articles|animal|animals|book|books|card|cards|drug|drugs|drink|drinks|film|films|food|foods|game|games|instrument|instruments|language|languages|movie|movies|novel|novels|play|plays|product|products|plant|plants|ship|ships|system|systems|symbol|symbols|trial|trials|train|trains|weapon|weapons|sword|swords|word|words) (is|was|were|are|do|did|does) $+)|(( which | what ) $? ( dog|cat|deer ) $*)|(( which | what ) $? part $+)|($ * what & * $ ? kind $*)|($ * ( composed | made ) & * $ ? ( from | through | using | by | of ) $*)|($ * what $* called $*)|($ * novel $*)|($ * ( thing | instance | object ) $*)|($ * ( which | what ) relative $+)|($ * ( which | (what (is|was|were|are | 's|does|did|do)|what's) ) $+ (to|in|on|at|of|for) $?)|($+ in (what|which) $+)|($ * ( which | what ) $* ( organization | trust | company ) $*)|($ * (what (was |is |are| 's)|what's) $* ( term | surname | address | name ) $*)|($ * (what|which) $* ( term | surname | address | name ) $* (do|did|does))|($ * (what (was |is |are| 's)|what's) $+ that $+)|(in what $+ was $+)|(what (is |are | was|were) the? first $+)|(what (is|was) the? alternate to $+)|(what (is |are | was|were) &? (a|an|the) fear of $+)|(( which | (what|what's) ) $? (was|is|are | 's|has|have|were|are) $* (first|largest|smallest|biggest|most|slowest|highest|last|longest|easiest) $*)|(how (do|does) (you|it) say `` $+ '' in $+)|(what does $+ collect $*)|(what is the name of $+)|($+ is known as what &?)|(what (did|does|do) $+ (call|calls|called) $*)|($+ (call|calls|called) $* what $*)|((what|how) (does|do) $+ call (an|a) $+)|((what (is|was|were|are | 's)|what's) $* color $+)|((what (is|was|were|are | 's)|what's) $? made of $*)|(what (color|product) (is |are | was|were|of|for) $+)|(what is the best way $+)|(what gaming devices $+)|(what $+ has the? extension $+)|(what is the? plural of $+)|(what $+ languages $+)|(what types of $+ (is |are | was) $+)|(what $+ can $+ but (can 't|can't) $+)|(what (keeps|keep|make) $+ (in|on|at) $+)|(on what $+)|((what|which) ${0,2} (do|does|did) $+ (eat|ride|study) $+)|(what does $+ when $+))"
    # ruleString = "((a|(c|d)?g?|e)?f)?"
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)
    #
    # ruleString = "$* ( a b c ) + $*"
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)
    #
    # ruleString = '(($ * abbreviation ( & | $ ) *)|($ * what & * $ ? does & * $ * ( stand for ) ( & | $ ) *))'
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)
    #
    # ruleString = '($ * abbreviation ( & | $ ) *)|($ * what & * $ ? does & * $ * ( stand for ) ( & | $ ) *)'
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)
    #
    #
    # ruleString = '( a )'
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)
    #
    # ruleString = '($ * abbreviation ( & | $ ) *)'
    # parseResult = ruleParser(ruleString)
    # print(parseResult)
    # reversed_re = reverse_regex(parseResult)
    # print(reversed_re)

    ruleString = '($ * abbreviation ( & | $ ) * a{1, 3} )'
    parseResult = ruleParser(ruleString)
    print(parseResult)
    reversed_re = reverse_regex(parseResult)
    print(reversed_re)