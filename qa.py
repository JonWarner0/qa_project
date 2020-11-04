import os
import sys
import nltk 
from nltk.corpus import wordnet as wn
import nltk.data
import re

# nltk & Wordnet quick reference
# https://www.nltk.org/howto/wordnet.html

""" 
-------- Global Definitions and Setup -------- 
"""
sent_detector = None # Extract sentences
try:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
except Exception:
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import spacy #TODO -> works on mine. Verify bash cmds for all
dependency_parser = spacy.load('en_core_web_sm') #TODO remove onese resolved
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
# regex for tagging to rid tenses
subj_re = re.compile(r'[a-z]*subj[a-z]*')
dobj_re = re.compile(r'[a-z]*dobj[a-z]*')
idobj_re = re.compile(r'[a-z]*pdobj[a-z]*')
noun_re = re.compile(r'PROPN | NOUN | NUM')  #Nouns and Nums together for simlicity
verb_re = re.compile(r'VERB') # TODO add AUX or make own with questions words?
question_word = re.compile(r'who|what|when|where|how|why')


""" 
-------- Class Definitions -------- 
"""
class qa:
    ID = ""
    question = ""
    ans = None
    dif = ""
    dep_parse = None 
    target = None 

    def __init__(self, ID, q, ans=None, dif=None):
        self.ID = ID
        self.question = q
        self.ans = ans
        self.dif = dif
        self.dep_parse = dependency_parser(self.question)
        self.target = composition(self.dep_parse)


class story:
    title = ""
    ID = ""
    date = ""
    corpus = None #list of sentences
    dep_parse = None 
    targets = []

    def __init__(self, title, ID, date, corpus):
        self.title = title
        self.ID = ID
        self.date = date
        self.corpus = sent_detector.tokenize(corpus) 
        self.dep_parse = [dependency_parser(s) for s in self.corpus]
        self.targets = [composition(s) for s in self.dep_parse]


class composition:
    """ Takes a spacy.Doc arguement and extracts needed information """
    sent = ''
    subj = ''
    dobj = ''
    idobj = ''
    nouns = []
    verbs = set()
    q_tag = None  #word & attachment -> predict answer
    #adj = [] # add later for descriptors etc.
    def __init__(self, sent):
        self.sent = sent.text
        for t in sent:
            if question_word.search(t.lemma_):
                self.q_tag = (t.lemma_, t.dep_)
            if subj_re.search(t.dep_):
                self.subj = t.lemma_
            if dobj_re.search(t.dep_):
                self.dobj = t.lemma_
            elif idobj_re.search(t.dep_):
                self.idobj = t.lemma_
            if verb_re.search(t.pos_): 
                self.verbs.add(t.lemma_)
            elif noun_re.search(t.pos_):
                self.nouns.append(t.lemma_)


""" 
-------- QA algorithm -------- 
"""
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


def generateAnswer(story, question):
    """ Computes overlap of root words. Includes stop words at this time """
    #TODO -> Add hypernyms and NER:(spacy.Doc).entity
    candidates = []
    for comp in story.targets: #gather candidate with overlapping subj and verbs(50%)
        if comp.subj == question.target.subj:
            verb_overlap = (comp.verbs).intersection(question.target.verbs)
            if (len(verb_overlap)) / (len(comp.verbs)+1) >= 0.5: #mismatch threshold
                candidates.append(comp)
    if len(candidates) == 0:
        return ''
    return candidates[0].sent

# Most likely parses
# What -> PRON : dobj | nsubj
# How | Where | When -> ADV : advmod
# Who -> PRON : nsubj


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
    for current_file in files: # Begin generating answers
        story = extractStory(directory+current_file+".story")
        questions = extractQuestions(directory+current_file+".questions")
        for q in questions:
            q.ans = generateAnswer(story, q)
            all_questions_answers.append(q)
    outputResponseFile(os.getcwd(), all_questions_answers)
