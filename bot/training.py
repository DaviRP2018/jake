import random

import nltk
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient

# nltk.download('punkt')
# nltk.download('wordnet')


class ChatbotTrainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ["?", "!", ".", ","]

    def load_intents(self):
        with MongoClient("localhost", 27017) as client:
            jake_db = client.jake_database
            intents_collection = jake_db.intents
            intents = list(intents_collection.find())
        return intents

    def load_data_from_mongo(self, intents):
        for intent in intents:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent["tag"]))

                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])

    def preprocess_data(self):
        self.words = [
            self.lemmatizer.lemmatize(word)
            for word in self.words
            if word not in self.ignore_letters
        ]
        self.words = sorted(set(self.words))

    def prepare_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [
                self.lemmatizer.lemmatize(word.lower())
                for word in word_patterns
            ]

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        self.train_x = np.array([i[0] for i in training])
        self.train_y = np.array([i[1] for i in training])

    def build_and_train_model(self):
        model = Sequential()
        model.add(
            Dense(128, input_shape=(len(self.train_x[0]),), activation="relu")
        )
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]), activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.FalseNegatives(),
            ],
        )

        hist = model.fit(
            np.array(self.train_x),
            np.array(self.train_y),
            epochs=600,
            batch_size=5,
            verbose=1,
        )

        model.save("chatbotmodel.keras", hist)

    def run_training(self):
        print("Training Chatbot Model...")
        intents = self.load_intents()
        self.load_data_from_mongo(intents)
        self.preprocess_data()
        self.prepare_training_data()
        self.build_and_train_model()
        print("Chatbot Model Trained Successfully!")
