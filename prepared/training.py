import random
import numpy as np
import pickle
import json

import tensorflow as tf

#Tokenization and Lemmatization(Tokenizes the patterns into words and lemmatizes them to reduce words to their base form)
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('E:/python__/chatbot/prepared/intents.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

#Saving Preprocessed Data (Stores the processed words and classes in pickle files.)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Model Training
#1(Preparing Training Data: Creates training data by converting the documents into bag-of-words arrays 
# and their corresponding intent classes into one-hot encoded arrays)
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
            
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])
    training.append([bag,output_row])
    
random.shuffle(training)
training = np.array(training, dtype=object)


train_x = np.array(training[:, 0].tolist())
train_y = np.array(training[:, 1].tolist())

#2(Building the Neural Network: Defines and trains a neural network using TensorFlow and Keras)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print('Done')
