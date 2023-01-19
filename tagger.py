import os
import io
import sys
import math
import collections
from nltk.tokenize import word_tokenize

TAGS = ('NOUN', 'PRONOUN', 'VERB', 'ADJECTIVE', 'ADVERB', 'CONJUNCTION', 'PREPOSITION', 'DETERMINER',
        'X','PUNCT', 'NUMBER')
pi_smooth = 1e-10
a_smooth = 1e-10
b_smooth = 1e-10
class Tagger:
    def __init__(self):
        """ Initialize class variables here """

        self.pi = {}
        self.a = {}
        self.b = {}
        for t in TAGS:
            self.pi[t] = 0
            self.a[t] = {}
            for t2 in TAGS:
                self.a[t][t2] = 0
            self.b[t] = collections.defaultdict(int)

    def load_corpus(self, path):
        """
        Returns all sentences as a sequence of (word, tag) pairs found in all
        files from as directory
        `path`.

        Inputs:
            path (str): name of directory
        Outputs:
            word_tags: 2d-list that represent sentences in the corpus. Each
            sentence is then represented as a list of tuples (word, tag)
        """
        if not os.path.isdir(path):
            sys.exit("Input path is not a directory")

        word_tags = []
        for filename in os.listdir(path):
            # Iterates over files in directory
            f = open(os.path.join(path, filename))
            for line in f.readlines():
                """ YOUR CODE HERE """
                if line == "\n":
                    continue
                sentence = []
                for string in line.split():
                    sentence.append(tuple(string.split("/")))
                word_tags.append(sentence)
        return word_tags


    def initialize_probabilities(self, sentences):
        """
        Initializes the initial, transition and emission probabilities into
        class variables

        Inputs:
            sentences: a 2d-list of sentences, usually the output of
            load_corpus
        Outputs:
            None
        """
        if type(sentences) != list:
            sys.exit("Incorrect input to method")

        """ 1. Compute Initial Tag Probabilities """
        for sentence in sentences:
            if sentence:
                (_, tag) = sentence[0]
                self.pi[tag] += 1

            for i in range(1, len(sentence)):
                (_, tag) = sentence[i]
                (_, prev_tag) = sentence[i - 1]
                self.a[prev_tag][tag] += 1

            for (token, tag) in sentence:
                self.b[tag][token] += 1

        """ 2. Compute Transition Probabilities """
        total = 0.0
        for t in TAGS:
            total += self.pi[t] + pi_smooth
        for t in TAGS:
            self.pi[t] = float(self.pi[t] + pi_smooth)/total

        """ 3. Compute Emission Probabilities """
        for t in TAGS:
            total = 0.0
            for t2 in TAGS:
                total += self.a[t][t2] + a_smooth
            for t2 in TAGS:
                self.a[t][t2] = float(self.a[t][t2] + a_smooth)/total

        for t in TAGS:
            total = b_smooth
            for token in self.b[t]:
                total += self.b[t][token]
            for token in self.b[t]:
                self.b[t][token] = float(self.b[t][token] + b_smooth)/total
            self.b[t]["<UNK>"] = b_smooth/total

        return


    def viterbi_decode(self, sentence):
        """
        Implementation of the Viterbi algorithm

        Inputs:
            sentence (str): a sentence with N tokens, be those words or
            punctuation, in a given language
        Outputs:
            likely_tags (list[str]): a list of N tags that most likely match
            the words in the input sentence. The i'th tag corresponds to
            the i'th word.
        """

        if type(sentence) != str:
            sys.exit("Incorrect input to method")

        """ Tokenize sentence """
        tokens = word_tokenize(sentence)

        """ Implement the Viterbi algorithm """
        delta = [[0.0 for _ in TAGS] for _ in tokens]
        back = [[0 for _ in TAGS] for _ in tokens]

        if not tokens:
            return []

        for i in range(len(TAGS)):
            t = TAGS[i]
            b = self.b[t][tokens[0]] if tokens[0] in self.b[t] \
                else self.b[t]["<UNK>"]
            delta[0][i] = self.pi[t] * b

        for t in range(1, len(tokens)):
            for j in range(len(TAGS)):
                max_i = 0
                max_val = -1
                for i in range(len(TAGS)):
                    tag_i = TAGS[i]
                    tag_j = TAGS[j]
                    val = delta[t - 1][i] * self.a[tag_i][tag_j]

                    if val > max_val:
                        max_val = val
                        max_i = i
                back[t][j] = max_i
                tag_j = TAGS[j]
                b = self.b[tag_j][tokens[t]] if tokens[t] in self.b[tag_j] \
                    else self.b[tag_j]["<UNK>"]
                delta[t][j] = max_val * b

        max_i = 0
        max_val = -1
        ret = []
        for i in range(len(TAGS)):
            if delta[-1][i] > max_val:
                max_val = delta[-1][i]
                max_i = i
        ret.append(TAGS[max_i])
        last = max_i
        for t in range(len(tokens) - 2, -1, -1):
            last = back[t + 1][last]
            ret.append(TAGS[last])

        return list(reversed(ret))

        return likely_tags

if __name__ == "__main__":
    tg = Tagger()
    tg.__init__()
    corpus_path = sys.argv[1]
    sentence = sys.argv[2]
    tg.initialize_probabilities(tg.load_corpus((corpus_path))) # "C:\\Users\\yeshw\\Downloads\\Yeshwanth\\new_NLP\\Assignment\\train\\modified_brown"
    # with open("C:\\Users\\yeshw\\Downloads\\Yeshwanth\\new_NLP\\Assignment\\30769880.txt") as file:
    #     for line in file:
    #         print(tg.viterbi_decode(line))
    # print(tg.viterbi_decode("the planet Jupiter and its moons are in effect a mini solar system ."))
    # print(tg.viterbi_decode("computers process programs accurately ."))
    print(tg.viterbi_decode(sentence))

