import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

#Repeated words are read as a single item.
wordlemm = WordNetLemmatizer()
#Reads as text file.
intents = json.loads(open('intents.json').read())

words = []
classes = []
docs = []
ignore_letters = [".", "!", "?", ","]

#This is separating all our patterns in the json.
for intent in intents["intent"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

#This is separating each word.
words = [wordlemm.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

#Setting individual words with numerical values by 0 or 1.
for document in docs:
    bag = []
    word_patterns = document[0]
    word_patterns = [wordlemm.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#This is creating our network layer in our model
#Dense layer does the below operation on the input and return the output
#Dropout is a technique used to prevent a model from overfitting.
model = Sequential()
#Relu, output the input directly if it is positive, otherwise, it will output zero.
#This behavior allows you to detect variations of attributes. It is used to find the best features considering their correlation.
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
#Softmax, converts a vector of numbers into a vector of probabilities
model.add(Dense(len(train_y[0]), activation = "softmax"))

sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save("chatbot_AI.h5", hist)








