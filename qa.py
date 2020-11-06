import os
import sys
import nltk 
from nltk.corpus import wordnet as wn
import nltk.data
import re


""" 
-------- Global Definitions and Setup/Downloads -------- 
"""
sent_detector = None # Sentence boundaries
try:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
except Exception:
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


dependency_parser = None
try:
    import spacy    # Dependency parsing install
    dependency_parser = spacy.load('en_core_web_sm') 
except Exception:
    os.system('pip3 install -U spacy')
    os.system('python3 -m spacy download en_core_web_sm')
    import spacy
    dependency_parser = spacy.load('en_core_web_sm')

# regex extracting tags
subj_re = re.compile(r'[a-z]*subj[a-z]*')
dobj_re = re.compile(r'[a-z]*dobj[a-z]*')
idobj_re = re.compile(r'[a-z]*pdobj[a-z]*')
noun_re = re.compile(r'PROPN | PRON | NOUN | NUM')  #Nouns and Nums together for simplicity
verb_re = re.compile(r'VERB') # TODO add AUX or make own with questions words?
adj_re = re.compile(r'ADJ')
root_re = lambda k: re.compile(r'[a-z]*{}[a-z]*'.format(k))
punc_re = re.compile(r"[\w']+|[.,!?;]") # findall to match dep_parse indexing
question_word = re.compile(r'who|what|when|where|how|why')


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


class story:
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
        self.ner = {t.text for t in sent.ents} #add .label_ for ORG, etc..
        self.nouns = set()
        self.no_lem_verb = dict()
        self.noun_phrase = {t.text for t in sent.noun_chunks}
        self.verbs = set()
        self.adj = set()
        self.score = 0
        for t in sent:
            if question_word.search(t.lemma_):
                self.q_tag = t.lemma_
            if subj_re.search(t.dep_):
                self.subj = t.lemma_
            if dobj_re.search(t.dep_):
                self.dobj = t.lemma_
            elif idobj_re.search(t.dep_):
                self.idobj = t.lemma_
            if verb_re.search(t.pos_): 
                self.verbs.add(t.lemma_)    #word was hyphenated and cant be found
                r = root_re(t.lemma_).search(t.text)
                if r:
                    self.no_lem_verb[t.lemma_] = punc_re.findall(self.sent).index(r.group(0))
            elif noun_re.search(t.pos_):
                self.nouns.add(t.lemma_)
            elif adj_re.search(t.pos_):
                self.adj.add(t.lemma_)


""" 
-------- QA algorithm -------- 
"""
# Most likely pos answer?
# What -> PRON : dobj | nsubj
# How | Where | When -> ADV : advmod
# Who -> PRON : nsubj

def getAnswer(story,question):
    verb_candidates = verbMatching(story,question)
    noun_candidates = nounMatching(story,question)
    #return max(story.target, key=lambda k:k.score)#for when scoring works right
    if len(verb_candidates) > 0:
        return verb_candidates
    return noun_candidates

def verbMatching(story, question): # FOR VERBED QUESTIONS
    candidate_sents = []
    for target in story.targets:
        syns_v, hyper_v = getSynsetsAndHypernyms(target.verbs, wn.VERB)
        s = syns_v.union(hyper_v)
        for v in s:
            syns_q_v, hyper_q_v = getSynsetsAndHypernyms(question.target.verbs, wn.VERB)
            q = syns_q_v.union(hyper_q_v)
            if v in q: 
                candidate_sents.append((target, v))
    candidate_phrase = []
    for target, verb in candidate_sents:
        if verb in target.no_lem_verb.keys():
            #weird indexing mapping to get the correct dependency parse
            idx = target.no_lem_verb[verb]
            sent_idx = story.targets.index(target)
            sent = story.dep_parse[sent_idx] 
            # TODO Extracts phrase based on dependencies
            # Currently not traveling far enough. need a NP conditional stop
            #left = [t.text for t in sent[idx].lefts]
            #right = [t.text for t in sent[idx].rights]
            #candidate_phrase.append(' '.join(left) +  ' '.join(right))
            target.score += 1
            candidate_phrase.append(sent.text)
    if len(candidate_phrase) == 0:
        return ''
    return candidate_phrase[0]


def nounMatching(story, question):
    candidates = set()
    for comp in story.targets: 
        ner_overlap = computeOverlap(comp.ner,question.target.ner)
        noun_overlap = computeOverlap(comp.nouns,question.target.nouns)
        np_overlap = computeOverlap(comp.noun_phrase, question.target.noun_phrase)
        comp.score += ner_overlap + np_overlap + noun_overlap
        candidates.add(comp)
    if len(candidates) == 0:
        return ''
    return max(candidates, key=lambda k:k.score).sent


def computeOverlap(s1,s2):
    ex1 = {s for t in s1 for s in t.split()}
    ex2 = {s for t in s2 for s in t.split()}
    return len(ex1.intersection(ex2))


def getSynsetsAndHypernyms(words, pos_):
    """ Uses WordNet to gather the synonyms-len(3) and hypernyms-len(9)"""
    synonyms = words.copy()
    hypernyms = set()
    counter = 0 # limit synonyms and hypernyms
    for w in words:
        for syns in wn.synsets(w, pos=pos_):
            if counter > 6: # take only top 6 synonyms 
                break
            synonyms.add(syns.name().split('.')[0]) #hard coded for now bc im lazy
            for s in syns.hypernyms()[:3]:
                hypernyms.add(s.name().split('.')[0])
                for xs in s.hypernyms()[:2]:
                    hypernyms.add(xs.name().split('.')[0])
                    for xxs in xs.hypernyms()[:2]:
                        hypernyms.add(xxs.name().split('.')[0])
                        for xxxs in xxs.hypernyms()[:2]:  
                            hypernyms.add(xxxs.name().split('.')[0])
            counter += 1
    return synonyms, hypernyms


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
    story_ = story(title, ID, date, corpus)
    return story_


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


def outputResponseFile(path, answers):
    """ Generates the Response.txt file with questions and answers """ 
    with open(path+'/Response.txt', 'w') as file:
        for q_a in answers:
            file.write('QuestionID: {}\nAnswer: {}\n\n'.format(q_a.ID, q_a.ans))


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
    outputResponseFile(os.getcwd(), all_questions_answers)
