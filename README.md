# qa_project: Jonathan Warner

Running: Tested on Python 3.8.0
* Run QA-script.txt to download nltk and spacy with their respective libraries that are used
* Run 'python3 qa.py <input_file>' to run the program and answers are printed to standard out


(a) Used nltk for sentence detection (punkt tokenizer) and access to wordnet for synonyms and hypernyms
    	 https://www.nltk.org
    Used spacy for dependency parsing, pos tagging, NER, NP extraction, and stemming
    	 https://spacy.io
(b) One document takes about 3 seconds
(c) I am a one man team
(d) My program only extracts the sentence that is most likley to contain the answer and is based on the verbs that appear in the question and the story. Until I can make a good scoring procedure, the nouns are not considered unless there are no matching verbs.
