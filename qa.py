import sys
import nltk # write script to install nltk in grading env
#import numpy as np

class qa:
    ID = ""
    question = ""
    ans = ""
    dif = ""

    def __init__(self, _id, _q="",_ans="",_dif=""):
        self.ID = _id
        self.question = _q
        self.ans = _ans
        self.dif = _dif


def parseStory(path):
    return []


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
