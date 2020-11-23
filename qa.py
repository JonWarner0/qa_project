import os
import sys
import nltk 
from nltk.corpus import wordnet as wn
import spacy  
import re

""" 
-------- Global Definitions and Setup/Downloads -------- 
"""
sent_detector = None # Sentence boundaries
try:
    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
except Exception:
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

dependency_parser = spacy.load('en_core_web_sm') 

# regex extraction patterns
noun_re = re.compile(r'PROPN | PRON | NOUN | NUM')  #Nouns and Nums together for simplicity
verb_re = re.compile(r'VERB') 
punc_re = re.compile(r"[\w']+|[.,!?;]") 
question_word = re.compile(r'(?i)who|what|when|where|how|why')
how_re = re.compile(r'(?i)how')
digit_re = re.compile(r'\d+')

""" 
-------- Class Definitions -------- 
"""
class qa:
    def __init__(self, ID, q, ans=None, dif=None):
        self.ID = ID
        self.question = q
        self.ans = ans
        self.dif = dif
        self.dep_parse = dependency_parser(self.question)
        self.target = composition(self.dep_parse)


class Story:
    def __init__(self, title, ID, date, corpus):
        self.title = title
        self.ID = ID
        self.date = date
        self.corpus = sent_detector.tokenize(corpus) 
        self.dep_parse = [dependency_parser(s) for s in self.corpus]
        self.targets = [composition(s) for s in self.dep_parse]

    def reset_scores(self):
        for c in self.targets:
            c.score = 0


class composition:
    """ Takes a spacy.Doc arguement and extracts needed information """
    def __init__(self, sent):
        self.sent = sent.text[:-1]
        self.ner = {t.text for t in sent.ents} 
        self.nouns = set()
        self.noun_phrase = {t.text for t in sent.noun_chunks}
        self.verbs = set()
        self.score = 0
        
        for t in sent:
            if question_word.search(t.lemma_):
                self.q_tag = t.lemma_
            if verb_re.search(t.pos_): 
                self.verbs.add(t.lemma_)
            elif noun_re.search(t.pos_):
                self.nouns.add(t.lemma_)


""" 
-------- QA algorithm -------- 
"""
def getAnswer(story,question):
    verbMatching(story,question)
    nounMatching(story,question)
    top = max(story.targets, key=lambda k:k.score) 
    if top.score < 6: # Not enough there to effectively extract substing but likely contains answer
        return top.sent
    return subDepExtraction(top,question) # extract substring


def verbMatching(story, question): 
    """ Gets all verbs synonyms and hypernyms from the question and story. 
        Compares them against each other to determine the overlap. Adds +3 to the 
        score of a senetence for each of the occurrence."""
    for target in story.targets:
        syns_v, hyper_v = getSynsetsAndHypernyms(target.verbs, wn.VERB)
        s = syns_v.union(hyper_v)
        for v in s:
            syns_q_v, hyper_q_v = getSynsetsAndHypernyms(question.target.verbs, wn.VERB)
            q = syns_q_v.union(hyper_q_v)
            if v in q: 
                target.score += 3 # weight verbs heavily as they are more unique


def nounMatching(story, question):
    """ Computes the overlap of of the NER phrases and NP that appear in 
        the story and the question"""
    for comp in story.targets: 
        ner_overlap = computeOverlap(comp.ner,question.target.ner)
        noun_overlap = computeOverlap(comp.nouns,question.target.nouns)
        np_overlap = computeOverlap(comp.noun_phrase, question.target.noun_phrase)
        comp.score +=  np_overlap + ner_overlap + noun_overlap 


def computeOverlap(s1,s2):
    ex1 = {s for t in s1 for s in t.split()}
    ex2 = {s for t in s2 for s in t.split()}
    return len(ex1.intersection(ex2))


def getSynsetsAndHypernyms(words, pos_):
    """ Uses WordNet to gather the synonyms and hypernyms"""
    synonyms = words.copy()
    hypernyms = set()
    counter = 0 # limit synonyms and hypernyms
    for w in words:
        for syns in wn.synsets(w, pos=pos_):
            if counter > 6: # take only top 6 synonyms 
                break
            synonyms.add(syns.name().split('.')[0]) 
            for s in syns.hypernyms()[:3]: # traverse up the hypernym tree to find commonalities
                hypernyms.add(s.name().split('.')[0])
                for xs in s.hypernyms()[:2]:
                    hypernyms.add(xs.name().split('.')[0])
                    for xxs in xs.hypernyms()[:2]:
                        hypernyms.add(xxs.name().split('.')[0])
                        for xxxs in xxs.hypernyms()[:2]:  
                            hypernyms.add(xxxs.name().split('.')[0])
            counter += 1
    return synonyms, hypernyms


def subDepExtraction(comp, q):
    """ Extract the substring related to the subdependency parse of the verb. 
    Requires that the most likely sentence is passed in as composition object"""
    sub_trees = []
    for word in dependency_parser(comp.sent): #extract the subtrees based on verbs
        if word.dep_ in ('xcomp','ccomp','cc','relcl','prep','advcl'):
            sub_trees.append(''.join(w.text_with_ws for w in word.subtree))
        elif word.dep_ in ('conj','dobj','mark'):
            sub_trees.append(''.join(w.text_with_ws for w in word.head.subtree))
    if len(sub_trees) == 0: # no candidates to make smaller
        return comp.sent 
    elif len(sub_trees) == 1: # only one option
        return sub_trees[0]
    best = ('',0)
    stem_q = ' '.join(d.lemma_ for d in q.dep_parse)
    for tree in sub_trees:  # Compare overlap for most likely subtree
        stem_tree = ' '.join(w.lemma_ for w in dependency_parser(tree))
        overlap = computeOverlap(stem_tree, stem_q)
        if overlap > best[1]:
            best = (tree, overlap) 
    # filters to increase percision at small cost to recall
    if how_re.search(q.question) and digit_re.search(best[0]): 
        return digit_re.search(best[0]).group(0)               
    return best[0]


def extractStory(_path):
    """ Extract story info and return a story object """
    input_data = []
    with open(_path, "r") as path:
        input_data = [line for line in path]
    title = input_data[0].replace("HEADLINE: ","") # Extract labels
    date = input_data[1].replace("DATE: ", "")
    ID = input_data[2].replace("STORYID: ", "")
    index = input_data.index("TEXT:\n") + 2 # marker of text start
    corpus = " ".join([line.strip('\n') for line in input_data[index:]])
    return Story(title, ID, date, corpus)


def extractQuestions(_path):
    """ Create a list of qa objects to hold all question/answer data for output """ 
    questions = []
    with open(_path, "r") as path:
        q = []
        for line in path: # Extract labels
            if line == '\n':
                ID = q[0].replace("QuestionID: ", "").strip('\n')
                question = q[1].replace("Question: ", "").strip('\n')
                dif = q[2].replace("Difficulty: ", "")
                questions.append(qa(ID, question, None, dif))
                q = []
            else:
                q.append(line)
    return questions


def outputResponseFile(answers):
    """ Generates the Response.txt file with questions and answers """ 
    for q_a in answers:
       print('QuestionID: {}'.format(q_a.ID))
       print('Answer: {}\n'.format(q_a.ans))


if __name__ == "__main__":
    directory_files = []
    with open(sys.argv[1], "r") as input_file: # Extract all files
        for line in input_file:
            directory_files.append(line.strip('\n'))
    directory = directory_files[0]
    files = directory_files[1:]
    all_questions_answers = []
    # Begin generating answers
    for current_file in files: 
        story = extractStory(directory+current_file+".story")
        questions = extractQuestions(directory+current_file+".questions")
        # Send the questions to their respective solver
        for q in questions:
            q.ans = getAnswer(story,q)
            all_questions_answers.append(q)
            story.reset_scores()
    outputResponseFile(all_questions_answers)
