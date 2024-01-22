import pickle
import random

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient

ERROR_THRESHOLD = 0.25


lemmatizer = WordNetLemmatizer()
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.keras")

client = MongoClient("localhost", 27017)
jake_db = client.jake_database
intents_collection = jake_db.intents


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    result = ""
    for i in intents_json:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


print("Chatbot is up!")

try:
    intents_list = list(intents_collection.find())

    while True:
        message = input("")
        ints = predict_class(message)
        print(ints)
        res = get_response(ints, intents_list)
        print(res)

except KeyboardInterrupt:
    print("Exiting")
finally:
    client.close()
