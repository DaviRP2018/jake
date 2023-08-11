# importing the required modules.
import json
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# reading the json.intense file
intents = json.loads(open("intents.json").read())

# creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # and adding them to words list

        # associating patterns with respective tags
        documents.append((word_list, intent["tag"]))

        # appending the tags to the class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# storing the root words or lemma
words = [
    lemmatizer.lemmatize(word) for word in words if word not in ignore_letters
]
words = sorted(set(words))

# saving the words and classes list to binary files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# we need numerical values of the
# words because a neural network
# needs numerical values to work with
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [
        lemmatizer.lemmatize(word.lower()) for word in word_patterns
    ]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)

# Convert training list to numpy arrays
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])


# creating a Sequential machine learning model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# compiling the model
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
    np.array(train_x), np.array(train_y), epochs=600, batch_size=5, verbose=1
)

# saving the model
model.save("chatbotmodel.keras", hist)

# print statement to show the
# successful training of the Chatbot model
print("Yay!")
