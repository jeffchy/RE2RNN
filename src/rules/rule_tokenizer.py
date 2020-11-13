from pyparsing import Literal, Word, alphas, Optional, OneOrMore, Forward, Group, ZeroOrMore, Literal, Empty, oneOf, nums, ParserElement
from pydash import flatten_deep

ParserElement.enablePackrat()
# WildCards = oneOf("$ % &") + "<:>" + Word(alphas)
# LeafWord = WildCards | Word(alphas + "'`") |  Word(alphas) + "<:>" + Word(alphas)
# $ means words, % means numbers, & means punctuations
WildCards = oneOf("$ % &")
LeafWord = WildCards | Word(alphas + "'`")
# aaa+ aaa* aaa? aaa{0,3} aaa{2}
RangedQuantifiers = Literal("{") + Word(nums) + Optional(
    Literal(",") + Word(nums)) + Literal("}")
Quantifiers = oneOf("* + ?") | RangedQuantifiers
QuantifiedLeafWord = LeafWord + Quantifiers
# a sequence
ConcatenatedSequence = OneOrMore(QuantifiedLeafWord | LeafWord)
# syntax root
Rule = Forward()
# ( xxx )
GroupStatement = Forward()
QuantifiedGroup = GroupStatement + Quantifiers
# (?<label> xxx)
# TODO: We don't need quantified capture group, so no QuantifiedCaptureGroup. And it is not orAble, can only be in the top level of AST, so it is easier to process
CaptureGroupStatement = Forward()
# xxx | yyy
orAbleStatement = QuantifiedGroup | GroupStatement | ConcatenatedSequence
OrStatement = Group(orAbleStatement +
                    OneOrMore(Literal("|") + Group(orAbleStatement)))

GroupStatement << Group(Literal("(") + Rule + Literal(")"))
CaptureGroupStatement << Group(Literal("(") + Literal("?") + Literal("<") + Word(alphas) + Literal(">")+ Rule + Literal(")"))
Rule << OneOrMore(OrStatement | orAbleStatement | CaptureGroupStatement)
ruleParser = lambda ruleString: flatten_deep(
    Rule.parseString(ruleString).asList())

if __name__ == "__main__":
    # ruleString = "$ * ( can' t | can't ) & * $? talk (&|$)*)"
    ruleString = "(($* name $? (a|one|two|three|four|five|six|seven|eight|nine|ten) $+)|(what $* ( played | play | run | study | studied | patent ) $*)|($ * ( which | what ) $* ( team | group | groups | teams ) $*)|($* ( which | (what (is|was|were|are | 's)|what's) ) $? (article|articles|animal|animals|book|books|card|cards|drug|drugs|drink|drinks|film|films|food|foods|game|games|instrument|instruments|language|languages|movie|movies|novel|novels|play|plays|product|products|plant|plants|ship|ships|system|systems|symbol|symbols|trial|trials|train|trains|weapon|weapons|sword|swords|word|words) $*)|($* ( which | what ) $? (article|articles|animal|animals|book|books|card|cards|drug|drugs|drink|drinks|film|films|food|foods|game|games|instrument|instruments|language|languages|movie|movies|novel|novels|play|plays|product|products|plant|plants|ship|ships|system|systems|symbol|symbols|trial|trials|train|trains|weapon|weapons|sword|swords|word|words) (is|was|were|are|do|did|does) $+)|(( which | what ) $? ( dog|cat|deer ) $*)|(( which | what ) $? part $+)|($ * what & * $ ? kind $*)|($ * ( composed | made ) & * $ ? ( from | through | using | by | of ) $*)|($ * what $* called $*)|($ * novel $*)|($ * ( thing | instance | object ) $*)|($ * ( which | what ) relative $+)|($ * ( which | (what (is|was|were|are | 's|does|did|do)|what's) ) $+ (to|in|on|at|of|for) $?)|($+ in (what|which) $+)|($ * ( which | what ) $* ( organization | trust | company ) $*)|($ * (what (was |is |are| 's)|what's) $* ( term | surname | address | name ) $*)|($ * (what|which) $* ( term | surname | address | name ) $* (do|did|does))|($ * (what (was |is |are| 's)|what's) $+ that $+)|(in what $+ was $+)|(what (is |are | was|were) the? first $+)|(what (is|was) the? alternate to $+)|(what (is |are | was|were) &? (a|an|the) fear of $+)|(( which | (what|what's) ) $? (was|is|are | 's|has|have|were|are) $* (first|largest|smallest|biggest|most|slowest|highest|last|longest|easiest) $*)|(how (do|does) (you|it) say `` $+ '' in $+)|(what does $+ collect $*)|(what is the name of $+)|($+ is known as what &?)|(what (did|does|do) $+ (call|calls|called) $*)|($+ (call|calls|called) $* what $*)|((what|how) (does|do) $+ call (an|a) $+)|((what (is|was|were|are | 's)|what's) $* color $+)|((what (is|was|were|are | 's)|what's) $? made of $*)|(what (color|product) (is |are | was|were|of|for) $+)|(what is the best way $+)|(what gaming devices $+)|(what $+ has the? extension $+)|(what is the? plural of $+)|(what $+ languages $+)|(what types of $+ (is |are | was) $+)|(what $+ can $+ but (can 't|can't) $+)|(what (keeps|keep|make) $+ (in|on|at) $+)|(on what $+)|((what|which) ${0,2} (do|does|did) $+ (eat|ride|study) $+)|(what does $+ when $+))"
    parseResult = ruleParser(ruleString)
    print(parseResult)
