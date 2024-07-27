import random
import numpy as np
import pickle
import json

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('E:/python__/chatbot/prepared/intents.json').read())


# Ensure paths are correct
words = pickle.load(open('E:/python__/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('E:/python__/chatbot/classes.pkl', 'rb'))
model = tf.keras.models.load_model('E:/python__/chatbot/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Bag of Words(Converts input sentences into a bag-of-words representation to use as input for the model)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#Prediction and Response
#1(Predicting the Intent: Uses the trained model to predict the intent of a user input)
def predict_class(sentence):
    bow = bag_of_words(sentence)
    # model = tf.keras.load_model()
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in result:
        return_list.append({'intent': classes[r[0]],'probability': str(r[1])})
    return return_list

#2(Generating the Response: Maps the predicted intent to a response from the intents JSON structure)
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
        
print("Go! bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
