from timing import timing
from experiment_2 import Word2vecSNLIModel

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


class DeepW2VSNLIModel(Word2vecSNLIModel):

    def __init__(self, n_classes: int, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 30):
        super().__init__(n_classes, vector_size, window, min_count)

        self.keras_model = None

    def _evaluate(self, dataset):
        """ extract the input, evaluate the similarity and classify """
        premise, hypothesis, labels = self._extract_features(dataset, ignore_doubtable=True)
        premise = self._pulish_dataframe(premise, duplicates=False)
        hypothesis = self._pulish_dataframe(hypothesis, duplicates=False)

        embedded_premise = self.embedded_sentences(self.premise_model, premise)
        embedded_hypothesis = self.embedded_sentences(self.hypothesis_model, hypothesis)
        input_matrix = np.hstack((embedded_premise, embedded_hypothesis))

        # Now use the embedded shit to create tensor and do more shit with the deep-learning model

        labels = self.convert_labels(labels, self.classes).astype(int)
        int_labels = np.take(np.eye(self.classes), labels, axis=0)
        # input_matrix = input_matrix.reshape(input_matrix.shape[0], 10, int(input_matrix.shape[1] / 10))  # for lstm

        if self.training:
            self.__init_keras_model_single_inp(input_matrix.shape)
            self.keras_model.fit(input_matrix, int_labels, batch_size=256, epochs=30, verbose=1)
            # self.__init_keras_model_double_inp(input_matrix.shape)
            # self.keras_model.fit((embedded_premise, embedded_hypothesis), int_labels, batch_size=256, epochs=100, verbose=1)

        inp = self.keras_model.predict(input_matrix)
        # inp = self.keras_model.predict((embedded_premise, embedded_hypothesis))

        # inp = (inp == inp.max(axis=1)[:, None]).astype(int)  # no classifier
        # print(accuracy_score(int_labels, (inp == inp.max(axis=1)[:, None]).astype(int)))

        self._classify(inp, labels)

    def _classify(self, output, labels):
        """ Train the classifier or get the predictions for test classification """
        if self.training:
            self.classifier.fit(output, labels)
        else:
            preds = self.classifier.predict(output)
            self.accuracy = accuracy_score(preds, labels)
            self.confusion_matrix = metrics.confusion_matrix(labels, preds)

    def __init_keras_model_single_inp(self, input_shape):
        model = Sequential()

        # model.add(Input(shape=(input_shape[1], input_shape[2])))  # for lstm
        model.add(Input(shape=(input_shape[1],)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))

        if self.classes == 2:
            model.add(Dense(self.classes, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(self.classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        self.keras_model = model

    def __init_keras_model_double_inp(self, input_shape):
        InputA = Input(shape=(self.vector_size,))
        A = Dense(64, activation="relu")(InputA)
        A = Dense(128, activation="relu")(A)

        InputB = Input(shape=(self.vector_size,))
        B = Dense(64, activation="relu")(InputB)
        B = Dense(128, activation="relu")(B)

        combined = Concatenate()([A, B])

        C = Dense(256, activation="relu")(combined)
        C = Dense(256, activation="relu")(C)

        if self.classes == 2:
            C = Dense(self.classes, activation="sigmoid")(C)
            model = Model(inputs=[InputA, InputB], outputs=C)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            C = Dense(self.classes, activation="softmax")(C)
            model = Model(inputs=[InputA, InputB], outputs=C)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        self.keras_model = model
