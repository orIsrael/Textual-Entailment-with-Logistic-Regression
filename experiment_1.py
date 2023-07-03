from timing import timing
from baseSNLIModel import BaseSNLIModel

import numpy as np
import pandas as pd

from nltk import edit_distance
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SemanticModel(BaseSNLIModel):
    """
    This model trying to solve RTE using vectorizers to find cosine similarity or calculating edit-distance.
    """

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.training = True
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    @timing
    def train(self, jsonl_file_path: str):
        self.training = True
        self._load_dataset(jsonl_file_path)
        self.__evaluate()

    @timing
    def test(self, jsonl_file_path: str):
        self.training = False
        self._load_dataset(jsonl_file_path)
        self.__evaluate()

    def _extract_features(self, dataset, ignore_doubtable: bool = False):
        premise, hypothesis, labels = super()._extract_features(dataset, ignore_doubtable)
        return premise.to_numpy(), hypothesis.to_numpy(), labels.to_numpy()

    @timing
    def __evaluate(self):
        """ Training / Testing - extract the featrues, calculate the similarity and classifiy """
        premise, hypothesis, labels = self._extract_features(self._dataset)
        similarity_matrix = self.__use_cosine_similarity(premise, hypothesis)
        self._classify(similarity_matrix.reshape(similarity_matrix.shape[0], 1), labels)

        # another approuch - merge and use vectorizer to send the premise + hypothesis vector to the classifier
        # a, b, _ = self.__merge_vectorize_matrices(premise, hypothesis)
        # sentences_matrix = sp.hstack((a, b))
        # self._classify(sentences_matrix, labels)

    @timing
    def __use_cosine_similarity(self, premises, hypothesis):
        premises_matrix, hypothesis_matrix, size = self.__merge_vectorize_matrices(premises, hypothesis)
        return np.array([cosine_similarity(premises_matrix[i], hypothesis_matrix[i]) for i in range(size)])

    @timing
    def __use_edit_distance_similarity(self, premises, hypothesis):
        size = len(premises)
        return np.array([edit_distance(premises[i], hypothesis[i]) for i in range(size)])

    @timing
    def __double_vectorize_matrices(self, premise, hypothesis):
        """
        Using two different vectorizers to fit both premise and hypothesis differently
        by using only premises vocabulary.

        :param premise: a 1-dim ndarray of premises
        :param hypothesis: a 1-dim ndarray where each value j is the hypothesis of premise j
        :return: premise matrix and hypothesis matrix and the size of them
        """
        premise_vectorizer = self.vectorizer
        premise_matrix = premise_vectorizer.fit_transform(premise)

        hypothesis_vectorizer = TfidfVectorizer(vocabulary=premise_vectorizer.get_feature_names_out())
        hypothesis_matrix = hypothesis_vectorizer.fit_transform(hypothesis)

        return premise_matrix, hypothesis_matrix, len(premise)

    @timing
    def __merge_vectorize_matrices(self, premise, hypothesis):
        """
        Merge the arrays and fir the vectorizer by using the merged matrix.
        It'll create a vocabulary using both premises and hypothesis words.

        :param premise: a 1-dim ndarray of premises
        :param hypothesis: a 1-dim ndarray where each value j is the hypothesis of premise j
        :return: premise matrix and hypothesis matrix and the size of them
        """
        sentences = np.array([premise, hypothesis]).flatten()
        sentences_matrix = self.vectorizer.fit_transform(sentences)

        slen = int(sentences.shape[0] / 2)
        premise_matrix = sentences_matrix[:slen, :]
        hypothesis_matrix = sentences_matrix[slen:, :]

        return premise_matrix, hypothesis_matrix, slen
