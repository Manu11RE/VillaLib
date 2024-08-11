# This script has the objetive training the model of chat

# Import libraries
import random
import json
import os
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer #Lemmatizer -> corriendo = correr

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

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

nltk.download('punkt') #Tokenizer
nltk.download('wordnet') #Data packet
nltk.download('omw-1.4') #Translate

words = []
classes = []
documents = []
ignore_letters = ['?', '¿', '¡', '!', ',', '.', ';', '-']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Se lematizan las palabras (se reducen a su forma base) y se eliminan los caracteres ignorados. 
# Luego, se eliminan duplicados y se ordenan. Finalmente, las listas words y classes se guardan 
# usando pickle para uso futuro.

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparación de los datos para el apredizaje automático

# Si la palabra está presente en la lista lematizada word_pattenrs, se añade un 1 a la bag. 
# Esto indica la presencia de la palabra en el documento.
# En caso contrario, se añade un 0 a la bag, indicando la ausencia de la palabra.

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_pattenrs = document[0]
    word_pattenrs = [lemmatizer.lemmatize(word.lower()) for word in word_pattenrs]
    for word in words:
        bag.append(1) if word in word_pattenrs else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Se barajan los datos de entrenamiento y se convierten en un arreglo numpy. 
# train_x contiene las bolsas de palabras y train_y las etiquetas de salida.

random.shuffle(training)
training = np.asarray(training, dtype="object")
print(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Se define un modelo secuencial en Keras con tres capas densas:

# La primera capa tiene 128 unidades y utiliza la activación relu, seguida de una capa de Dropout para 
# reducir el sobreajuste.
# La segunda capa tiene 64 unidades, también con relu, seguida de otra capa de Dropout.
# La capa de salida utiliza softmax para clasificar en una de las clases.

# Se compila el modelo utilizando el optimizador SGD con una tasa de aprendizaje específica, decaimiento, 
# momentum y Nesterov. Finalmente, se entrena el modelo con 100 épocas y un tamaño de lote de 5, y se 
# guarda el modelo entrenado en un archivo chatbot_model.h5.

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Se utiliza el optimizador Stochastic Gradient Descent (SGD) con parámetros específicos para la tasa de 
# aprendizaje, la descomposición y el impulso de Nesterov.

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Entrenamiento del modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5", train_process)






