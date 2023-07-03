import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


class BaseSNLIModel(ABC):
    """ A base model for training and testing. Using SNLI dataset """

    def __init__(self, n_classes: int):
        self.classes = n_classes
        self._dataset = None
        self.classifier = LogisticRegression(max_iter=300)
        self.training = True

        self.accuracy = 0.0
        self.confusion_matrix = None

    @abstractmethod
    def train(self, jsonl_file_path: str):
        raise NotImplementedError("subclasses must override 'train' method in order to operate")

    @abstractmethod
    def test(self, jsonl_file_path: str):
        raise NotImplementedError("subclasses must override 'test' method in order to operate")

    def _load_dataset(self, jsonl_file_path: str):
        assert jsonl_file_path.endswith(".jsonl")
        self._dataset = pd.read_json(jsonl_file_path, lines=True)

    def _extract_features(self, dataset, ignore_doubtable: bool = False):
        """
        Extract all the feature this model uses

        :param dataset: A dataset represented as DataFrame
        :param ignore_doubtable: True to ignore doubtable labels (Where the label unkown)
        """
        if ignore_doubtable:
            dataset = dataset.drop(dataset[dataset['gold_label'] == '-'].index)
        hypothesis = dataset['sentence2']  # hypothesis
        premise = dataset['sentence1']  # original text
        labels = dataset['gold_label']  # relationship
        return premise, hypothesis, labels

    def _classify(self, sentences, labels):
        """ Train the classifier or get the predictions for test classification """
        labels = self.convert_labels(labels, self.classes)

        if self.training:
            self.classifier.fit(sentences, labels)
        else:
            preds = self.classifier.predict(sentences)
            self.accuracy = accuracy_score(preds, labels)
            self.confusion_matrix = metrics.confusion_matrix(labels, preds)

    @staticmethod
    def convert_labels(labels, n_way: int = 3):
        """ Converter to numerical labels """
        label_set = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 1}
        if n_way == 2:
            label_set['contradiction'] = 1
        return np.array([label_set[ll] for ll in labels], dtype=int)

    @staticmethod
    def split_dataset(dataset, test_ratio=0.2, start_offset_ratio: float = 0.0):
        assert dataset is not None
        test_samples = int(len(dataset) * test_ratio)
        start_offset = int(len(dataset) * start_offset_ratio)
        test_ds = dataset[start_offset:start_offset + test_samples]
        train_ds = dataset.drop(test_ds.index)
        return train_ds, test_ds
