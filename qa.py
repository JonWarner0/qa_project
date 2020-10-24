import os
import sys
import nltk # write script to install nltk in grading env
import spacy 
import re

class qa:
    ID = ""
    question = ""
    ans = ""
    dif = ""
    def __init__(self, ID, q, ans="", dif=""):
        self.ID = ID
        self.question = q
        self.ans = ans
        self.dif = dif


class story:
    title = ""
    ID = ""
    date = ""
    corpus = None
    def __init__(self, title, ID, date, corpus):
        self.title = title
        self.ID = ID
        self.date = date
        self.corpus = corpus


def parseStory(_path):
    input_data = []
    with open(_path) as path:
        input_data = [line for line in path]
    title = input_data[0].replace("HEADLINE: ","")
    date = input_data[1].replace("DATA: ", "")
    ID = input_data[2].replace("STORYID: ", "")
    index = input_data.index("TEXT:\n") + 2 # start of text
    story_o = story(title, ID, date, input_data[index:])
    pos_tags = [nltk.pos_tag(t) for t in story_o.corpus]
    return story_o, pos_tags


def parseQuestions(path):
    return []


def generateAnswers(story, questions):
    return []


if __name__ == "__main__":
    directory_files = []
    with open(sys.argv[1]) as input_file:
        for line in input_file:
            directory_files.append(line)
    directory = directory_files[0]
    files = directory_files[1:]
    for current_file in files:
        story = parseStory(directory+current_file+".story")
        questions = parseQuestions(directory+current_file+"questions")
        answers = generateAnswers(story, questions)
