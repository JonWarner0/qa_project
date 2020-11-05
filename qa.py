import os
import sys
import nltk 
from nltk.corpus import wordnet as wn
import nltk.data
import re

""" 
-------- Global Definitions and Setup -------- 
"""
sent_detector = None # Sentence boundaries
try:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
except Exception:
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import spacy #TODO -> works on mine. Verify bash cmds for all
dependency_parser = spacy.load('en_core_web_sm') #TODO remove once resolved
"""
dependency_parser = None
try:
    import spacy    # Dependency parsing install
    dependency_parser = spacy.load('en_core_web_sm') 
except Exception:
    os.system('pip3 install -U spacy')
    #os.system('python3 -m spacy download en_core_web_sm')
    import spacy
    dependency_parser = spacy.load('en_core_web_sm')
"""
# regex extracting tags
subj_re = re.compile(r'[a-z]*subj[a-z]*')
dobj_re = re.compile(r'[a-z]*dobj[a-z]*')
idobj_re = re.compile(r'[a-z]*pdobj[a-z]*')
noun_re = re.compile(r'PROPN | PRON | NOUN | NUM')  #Nouns and Nums together for simlicity
verb_re = re.compile(r'VERB') # TODO add AUX or make own with questions words?
adj_re = re.compile(r'ADJ')
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
        self.sent = sent.text
        self.ner = {t.text for t in sent.ents} #add .label_ for ORG, etc..
        self.nouns = set()
        self.noun_phrase = {s for t in sent.noun_chunks for s in t.text.split()}
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
                self.verbs.add(t.lemma_)
            elif noun_re.search(t.pos_):
                self.nouns.add(t.lemma_)
            elif adj_re.search(t.pos_):
                self.adj.add(t.lemma_)


""" 
-------- QA algorithm -------- 
"""
# Most likely parses
# What -> PRON : dobj | nsubj
# How | Where | When -> ADV : advmod
# Who -> PRON : nsubj

def who_what_questions(story, question):
    """ Get the synonyms and hypernyms of the verbs and nouns, compute they're
        overlap, then divide them by them by the questions target verbs
        (with +1 to avoid division by 0). By using the amount of overlap as 
        a score, the sentences can be ranked by most likely to contain the answer.
    """
    candidates = set()
    for comp in rootMatching(story,question): #gather candidate with overlapping subj and verbs(50%)
        ner_overlap = computeOverlap(comp.ner,question.target.ner)
        noun_overlap = computeOverlap(comp.nouns,question.target.nouns)
        np_overlap = computeOverlap(comp.noun_phrase,question.target.noun_phrase)
        if (ner_overlap or noun_overlap or np_overlap) > 0: # Constrain expanded
            comp.score += ner_overlap + np_overlap
            candidates.add(comp)
            s, s_hyper = getSynsetsAndHypernyms(comp.verbs, wn.VERB)
            q, q_hyper = getSynsetsAndHypernyms(question.target.verbs, wn.VERB)
            # Compute overlap
            verb_overlap = computeOverlap(q,s)
            hyper_overlap = computeOverlap(q_hyper,s_hyper)
            comp.score += verb_overlap + hyper_overlap
    if len(candidates) == 0:
        return ''
    return max(candidates, key=lambda k:k.score).sent


def how_where_when_question(story,question):
    candidates = set()
    for comp in rootMatching(story,question): #gather candidate with overlapping subj and verbs(50%)
        sv, sv_hyper = getSynsetsAndHypernyms(comp.verbs, wn.VERB)
        qv, qv_hyper = getSynsetsAndHypernyms(question.target.verbs, wn.VERB)
        s_adj, s_adj_hyper = getSynsetsAndHypernyms(comp.verbs, wn.ADJ)
        q_adj, q_adj_hyper = getSynsetsAndHypernyms(question.target.verbs, wn.ADJ)
        # Compute overlap
        overlap = []
        overlap.append(computeOverlap(qv,sv))
        overlap.append(computeOverlap(qv_hyper, sv_hyper))
        overlap.append(computeOverlap(q_adj,s_adj))
        overlap.append(computeOverlap(q_adj_hyper,s_adj_hyper))
        overlap.append(computeOverlap(comp.ner,question.target.ner))
        overlap.append(computeOverlap(comp.nouns, question.target.nouns))
        comp.score += sum(overlap)
        candidates.add(comp)
    if len(candidates) == 0:
        return ''
    return max(candidates, key=lambda k:k.score).sent


def computeOverlap(s1,s2):
    return len(s1.intersection(s2))


def rootMatching(story, question):
    candidates = []
    q_roots = {chunk.root.head.lemma_ for chunk in question.dep_parse.noun_chunks}
    for i in range(len(story.dep_parse)):
        roots = {chunk.root.head.lemma_ for chunk in story.dep_parse[i].noun_chunks}
        s, hyper = getSynsetsAndHypernyms(roots, wn.VERB)
        q, q_hyper = getSynsetsAndHypernyms(q_roots, wn.VERB)
        roots = s.union(hyper)
        q_roots = q.union(q_hyper)
        if computeOverlap(q_roots, roots) > 0:
            candidates.append(story.targets[i])
    return candidates


def getSynsetsAndHypernyms(words, pos_):
    """ Uses WordNet to gather the synonyms-len(3) and hypernyms-len(9)"""
    synonyms = words.copy()
    hypernyms = set()
    counter = 0 # limit synonyms and hypernyms
    for w in words:
        for syns in wn.synsets(w, pos=pos_):
            if counter > 2: # take only top 3 synonyms 
                break
            synonyms.add(syns.name().split('.')[0])
            for s in syns.hypernyms()[:2]:
                hypernyms.add(s.name().split('.')[0])
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
                question = q[1].replace("Question: ", "")
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
            if q.target.q_tag == ('who' or 'what'):
                q.ans = who_what_questions(story, q)
            else:
                q.ans = how_where_when_question(story, q)
            all_questions_answers.append(q)
            story.reset_scores()
    outputResponseFile(os.getcwd(), all_questions_answers)
