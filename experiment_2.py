from timing import timing
from baseSNLIModel import BaseSNLIModel

import numpy as np
import pandas as pd

import nltk
from gensim.models.word2vec import Word2Vec

from nltk.corpus import stopwords
# from nltk import PorterStemmer
# from nltk.tokenize import word_tokenize - no improvement. slower.


class Word2vecSNLIModel(BaseSNLIModel):
    """
    This model uses word2vec in order to predict the similarity between premise and hypothesis
    """

    def __init__(self, n_classes: int, vector_size: int = 100, window: int = 5, min_count: int = 1):
        super().__init__(n_classes)

        self.download_resources()

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

        self.premise_model = None
        self.hypothesis_model = None
        self.vocab = None

        self.stop_words = set(stopwords.words('english'))

    def _pulish_dataframe(self, df, duplicates=True, lower=True, tokenize=True):
        """
        Remove duplicates, lower characters and tokenize each sentence to list of words

        :param df: Pandas Series or DataFrame with single column, where each item is a sentence.
        :return: List of lists - each item in the list is list of words.
        """
        if duplicates:  # remove duplicates
            df = df.drop_duplicates()
        if lower:  # to-lower
            df = df.str.lower()
        if tokenize:  # tokenize sentences
            # df = [[w for w in s.split() if w.lower() not in self.stop_words] for s in df]  # decrease accuracy.
            df = [s.split() for s in df]
        return df

    @timing
    def train(self, jsonl_file_path: str):
        self.training = True
        self._load_dataset(jsonl_file_path)
        self.__training_word2vec()
        self._evaluate(self._dataset)

    def test(self, jsonl_file_path: str):
        self.training = False
        self._load_dataset(jsonl_file_path)
        self._evaluate(self._dataset)

    @timing
    def __training_word2vec(self):
        """ Train 2 Word2Vec models for both premises and hypothesis. And set the vocabulary """
        premise, hypothesis, _ = self._extract_features(self._dataset, True)
        premise = self._pulish_dataframe(premise)
        hypothesis = self._pulish_dataframe(hypothesis)

        self.premise_model = Word2Vec(premise, sg=1,
                                      vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        self.hypothesis_model = Word2Vec(hypothesis, sg=1,
                                         vector_size=self.vector_size, window=self.window, min_count=self.min_count)

    @timing
    def embedded_sentences(self, w2v_model, sentences: str):
        return np.vstack([self.embedded_sent(w2v_model, s) for s in sentences])

    def embedded_sent(self, w2v_model, sentence: str):
        emb = np.array([w2v_model.wv[word] for word in sentence if word in w2v_model.wv])
        if emb.shape[0] != 0:
            return emb.mean(axis=0)
        return np.zeros(self.vector_size)

    @timing
    def cos_similarity(self, p, h):
        return np.array([x.dot(y.transpose()) / (np.linalg.norm(x) * np.linalg.norm(y)) for x, y in zip(p, h)])

    def _evaluate(self, dataset):
        """ extract the input, evaluate the similarity and classify """
        premise, hypothesis, labels = self._extract_features(dataset, ignore_doubtable=True)
        premise = self._pulish_dataframe(premise, duplicates=False)
        hypothesis = self._pulish_dataframe(hypothesis, duplicates=False)

        embedded_premise = self.embedded_sentences(self.premise_model, premise)
        embedded_hypothesis = self.embedded_sentences(self.hypothesis_model, hypothesis)
        input_matrix = np.hstack((embedded_premise, embedded_hypothesis))

        # input_matrix = []
        # for i in range(len(premise)):
        #     embedded_premise = self.embedded_sent(self.premise_model, premise[i])
        #     embedded_hypothesis = self.embedded_sent(self.hypothesis_model, hypothesis[i])
        #     input_matrix.append(np.hstack((embedded_premise, embedded_hypothesis)))
        # input_matrix = np.vstack(input_matrix)

        self._classify(input_matrix, labels)

    def set_vocab(self, vocabs):
        """
        Init the vocabulary for this model

        :param vocabs: A list of vocabularies. (list of list of words)
        """
        self.vocab = {word for voc in vocabs for sent in voc for word in sent}

    @staticmethod
    def download_resources():
        """ Download all the reasources this model need. """
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            nltk.download('stopwords')
