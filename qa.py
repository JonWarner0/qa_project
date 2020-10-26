import os
import sys
import nltk # write script to install externals in grading env
from nltk import word_tokenize
from nltk.stem import PorterStemmer as ps
import re
from nerd import ner


class qa:
    ID = ""
    question = ""
    ans = None
    dif = ""
    pos_tags = None 
    root_entity = None 
    def __init__(self, ID, q, ans=None, dif=None):
        self.ID = ID
        self.question = q
        self.ans = ans
        self.dif = dif
        self.pos_tags = nltk.pos_tag(word_tokenize(q))
        self.root_entity = [ps().stem(str(s)) for s in ner.name(q)]
        

class story:
    title = ""
    ID = ""
    date = ""
    corpus = None
    pos_tags = None
    root_entity = []
    def __init__(self, title, ID, date, corpus):
        self.title = title
        self.ID = ID
        self.date = date
        self.corpus = corpus
        self.pos_tags = [nltk.pos_tag(word_tokenize(t)) for t in self.corpus]
        for p in self.corpus:
            self.root_entity.append([ps().stem(s) for s in p])
        # currently paragraphs in corpus instead of single sentences.


def extractStory(_path):
    input_data = []
    with open(_path, "r") as path:
        input_data = [line for line in path]
    title = input_data[0].replace("HEADLINE: ","")
    date = input_data[1].replace("DATA: ", "")
    ID = input_data[2].replace("STORYID: ", "")
    index = input_data.index("TEXT:\n") + 2 # start of text
    corpus = list(filter('\n'.__ne__,input_data[index:]))
    corpus = [line.strip('\n') for line in corpus]
    story_o = story(title, ID, date, corpus)
    return story_o


def extractQuestions(_path):
    questions = []
    with open(_path, "r") as path:
        q = []
        for line in path:
            if line == '\n':
                ID = q[0].replace("QuestionsID: ", "")
                question = q[1].replace("Question: ", "")
                dif = q[2].replace("Difficulty: ", "")
                questions.append(qa(ID, question, None, dif))
                q = []
            else:
                q.append(line)
    return questions


def questionMorph(question):
    return []


def generateAnswer(story, question):
    candidates = []
    entities = set(question.root_entity)
    for i in range(len(story.root_entity)): # root word overlap
        candidates.append((i, set(story.root_entity[i]).union(entities)))
    return sorted(candidates, key=lambda k: len(k[1]), reverse=True)


if __name__ == "__main__":
    directory_files = []
    with open(sys.argv[1], "r") as input_file:
        for line in input_file:
            directory_files.append(line.strip('\n'))
    directory = directory_files[0]
    files = directory_files[1:]
    for current_file in files:
        story = extractStory(directory+current_file+".story")
        questions = extractQuestions(directory+current_file+".questions")
        for q in questions:
            answers = generateAnswer(story, q)
            print(answers[0])
        