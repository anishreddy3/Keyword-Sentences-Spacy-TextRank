## Implementing KeyWord Extraction
## import the required packages

from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from prettytable import PrettyTable
import glob2

nlp = spacy.load('en_core_web_sm')

class KeyWordRank():
    ## Extracting Keywords from text

    def __init__(self):
        self.d = 0.85  # damping co-efficient typically set around at 0.85
        self.min_diff = 1e-5 # Threshold for convergence
        self.steps = 10 # number of iteration steps
        self.node_weight = None # saving keywords along with its weight

    def imp_stopwords(self, stopwords):
        ## Set the stop_words
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sent_seg(self, doc, candidate_pos, lower):
        ## Store the words in candidate position
        sentences = []
        for sent in doc.sents:
            sel_words = []
            for token in sent:
                # Store Words only with candidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        sel_words.append(token.text.lower())
                    else:
                        sel_words.append(token.text)
            sentences.append(sel_words)
        return sentences

    def get_vocab(self, sentences):
        ## Gather every token
        vocabulary = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocabulary:
                    vocabulary[word] = i
                    i += 1
        return vocabulary

    def get_tok_pairs(self, window_size, sentences):
        ## Get all token pairs from windows in sentences
        tok_pairs =  list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in tok_pairs:
                        tok_pairs.append(pair)
        return tok_pairs

    def make_symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_mat(self, vocab, tok_pairs):
        ## Get the normalized Matrix
        # Build Matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word_a, word_b in tok_pairs:
            i, j = vocab[word_a], vocab[word_b]
            g[i][j] = 1

        # Symmetric Matrix
        g = self.make_symmetrize(g)

        # Norm matrix by column
        norm = np.sum(g, axis = 0)
        g_norm = np.divide(g, norm, where=norm!=0)

        return g_norm


    def get_keyword(self, text, number=10):
        ## Get the top Keywords along with their respective sentences and generate the table
        import glob, os
        doc = glob.glob('*.txt')
        x = PrettyTable() # Generate the Table
        x.field_names = ["Word(#)", "Documents", "Sentences containing the word" ] #Column names
        # Align all columns to the left side
        x.align["Word(#)"] = "l"
        x.align["Documents"] = "l"
        x.align["Sentences containing the word"] = "l"
        Dict = {} # Initialize the dictionary
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse = True))
        for i, (key, value) in enumerate(node_weight.items()):
            docs = [] # Documents Containing the word
            Dict[key] = ([sentence for sentence in text.replace("\n","").split('.') if key in sentence])

            if ('output_file.txt' in doc):
                doc.remove('output_file.txt')
            #print(len(doc))
            for j in range(len(doc)):

                file = open(doc[j],"r")

                contents = file.read()

                if key in contents:
                    docs.append(doc[j])

                file.close()

            x.add_row([key, set(docs), Dict[key]])
            if i > number:
                print(x)
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):


                ## Function to Analyze text

                ## Set all stop Words

                self.imp_stopwords(stopwords)

                # Parse text by spacy
                doc = nlp(text)

                ## Filter sentences
                sentences = self.sent_seg(doc, candidate_pos, lower)

                ## Build Vocabulary
                vocab = self.get_vocab(sentences)

                ## Get token pairs from windows
                tok_pairs = self.get_tok_pairs(window_size, sentences)

                ## Get normalized Matrix
                g = self.get_mat(vocab,tok_pairs)

                ## Initialization for weight
                pr = np.array([1] * len(vocab))

                # iteration
                previous = 0
                for epoch in range(self.steps):
                    pr = (1-self.d) + self.d * np.dot(g, pr)
                    if abs(previous - sum(pr)) < self.min_diff:
                        break
                    else:
                        previous = sum(pr)

                # Get weight for each node
                node_weight = dict()
                for word, index in vocab.items():
                    node_weight[word] = pr[index]

                self.node_weight = node_weight

## Read from all documents
filenames = glob2.glob('*.txt')
with open('output_file.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read() + '\n')


text = open('output_file.txt','r').read()
trank = KeyWordRank() ## Ranking Keywords
trank.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False) ## Analyze text
trank.get_keyword(text,8) ## Choosing the number of keywords, will display n+2 keywords along with their sentences
