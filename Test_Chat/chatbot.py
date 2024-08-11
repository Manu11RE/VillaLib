import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file
json_file_path = os.path.join(script_dir, 'test_intents.json')

# Attempt to read the JSON file
try:
    with open(json_file_path, 'r') as file:
        intents = json.load(file)
    # Process the intents data
except FileNotFoundError:
    print("Error: File 'test_intents.json' not found.")
except PermissionError:
    print("Error: Permission denied to open 'test_intents.json'.")
except json.JSONDecodeError:
    print("Error: Unable to parse JSON data in 'test_intents.json'.") 

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break

    return result

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)