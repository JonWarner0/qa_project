import os
import sys
import nltk 
from nltk import word_tokenize
from nltk.stem import PorterStemmer as ps
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
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
# t.text, t.tag_, t.head.text, t.dep_
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


""" 
-------- Class Definitions -------- 
"""
class qa:
    ID = ""
    question = ""
    ans = None
    dif = ""
    pos_tags = None 
    dep_parse = None 

    def __init__(self, ID, q, ans=None, dif=None):
        self.ID = ID
        self.question = q
        self.ans = ans
        self.dif = dif
        #self.pos_tags = nltk.pos_tag(word_tokenize(q))
        self.dep_parse = dependency_parser(self.question)


class story:
    title = ""
    ID = ""
    date = ""
    corpus = None #list of sentences
    pos_tags = None
    dep_parse = None 

    def __init__(self, title, ID, date, corpus):
        self.title = title
        self.ID = ID
        self.date = date
        self.corpus = sent_detector.tokenize(corpus) 
        #self.pos_tags = [nltk.pos_tag(word_tokenize(t)) for t in self.corpus]
        self.dep_parse = [dependency_parser(s) for s in self.corpus]


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


def getQuestionTarget(question):
    """ Rewrite with most likely sense and attempt to match corpus """
    #TODO -> Dependency parse and travel from VP to NP and check the respective
    # hypernyms and the hypernyms synsets for matching. Extract all NP's
    # Calculate cosine similarity of Sentence and Question (?)
    root_idx = 0
    expansion = []
    for t in question.dep_parse:
        expansion.append([t.text, t.tag_, t.head.text, t.dep_])
        if t.dep_ == 'ROOT':
            root_idx = len(expansion)-1
    
    return []


def generateAnswer(story, question):
    """ Computes overlap of root words. Includes stop words at this time """
    candidates = []
    return candidates


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
