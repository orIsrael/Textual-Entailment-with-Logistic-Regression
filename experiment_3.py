from timing import timing
from baseSNLIModel import BaseSNLIModel

import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk


class Doc2vecSNLIModel(BaseSNLIModel):
    """
    This model uses doc2vec in order to predict the similarity between premise and hypothesis
    """

    def __init__(self, n_classes: int, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 30):
        super().__init__(n_classes)

        self.download_resources()

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

        self.d2v_model = None

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @timing
    def __pulish_dataframe(self, df, duplicates=True, lower=True, tokenize=True, lemmatizer=True, stop_words=True):
        """
        Remove duplicates, lower characters and tokenize each sentence to list of words

        :param df: Pandas Series or DataFrame with single column, where each item is a sentence.
        :return: List of lists - each item in the list is list of words.
        """
        if duplicates:
            df = df.drop_duplicates()
        if lower:
            df = df.str.lower()
        if tokenize:
            df = [word_tokenize(s) for s in df]
            if stop_words:
                df = [[w for w in s if w is not self.stop_words] for s in df]
            if lemmatizer:
                df = [[self.lemmatizer.lemmatize(w) for w in s] for s in df]
        return df

    @timing
    def train(self, jsonl_file_path: str):
        self.training = True
        self._load_dataset(jsonl_file_path)
        self.__training_doc2vec()
        self.__evaluate(self._dataset)

    def test(self, jsonl_file_path: str):
        self.training = False
        self._load_dataset(jsonl_file_path)
        self.__evaluate(self._dataset)

    def _extract_features(self, dataset, ignore_doubtable: bool = False):
        premise, hypothesis, labels = super()._extract_features(dataset, ignore_doubtable)
        return premise + hypothesis, labels

    @timing
    def __training_doc2vec(self):
        """ Train 2 Word2Vec models for both premises and hypothesis. And set the vocabulary """
        premise_hypothesis, labels = self._extract_features(self._dataset, True)
        tokens = self.__pulish_dataframe(premise_hypothesis, duplicates=False)

        tags = [TaggedDocument(d, [i]) for d, i in zip(tokens, labels)]

        self.d2v_model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        self.d2v_model.build_vocab(tags)
        self.d2v_model.train(tags, total_examples=self.d2v_model.corpus_count, epochs=self.epochs)

    def __evaluate(self, dataset):
        """ extract the input, evaluate the similarity and classify """
        premise_hypothesis, labels = self._extract_features(dataset, ignore_doubtable=True)
        tokens = self.__pulish_dataframe(premise_hypothesis, duplicates=False)

        embedded_tokens = [self.d2v_model.infer_vector(s) for s in tokens]
        self._classify(embedded_tokens, labels)

    @staticmethod
    def download_resources():
        """ Download all the reasources this model need. """
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/omw-1.4.zip')
        except LookupError:
            nltk.download('omw-1.4')
