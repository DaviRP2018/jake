import pickle
import random

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient

ERROR_THRESHOLD = 0.25


class Bot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = pickle.load(open("words.pkl", "rb"))
        self.classes = pickle.load(open("classes.pkl", "rb"))
        self.model = load_model("chatbotmodel.keras")
        self.client = MongoClient("localhost", 27017)
        self.jake_db = self.client.jake_database
        self.intents_collection = self.jake_db.intents

    def clean_up_sentences(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [
            self.lemmatizer.lemmatize(word) for word in sentence_words
        ]
        return sentence_words

    def bagw(self, sentence):
        sentence_words = self.clean_up_sentences(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bagw(sentence)
        res = self.model.predict(np.array([bow]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [
            {"intent": self.classes[r[0]], "probability": str(r[1])}
            for r in results
        ]
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]["intent"]
        result = ""
        for i in intents_json:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result

    def run(self):
        try:
            print("Chatbot is up!")
            intents_list = list(self.intents_collection.find())

            while True:
                message = input("")
                ints = self.predict_class(message)
                print(ints)
                res = self.get_response(ints, intents_list)
                print(res)

        except KeyboardInterrupt:
            print("Exiting")
        finally:
            self.client.close()
