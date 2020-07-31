import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle
import speech_recognition as sr
import pyttsx3
language = 'en'
converter = pyttsx3.init()
converter.setProperty('rate',150)
converter.setProperty('volume',0.5)
def get_string():
    answer = ''
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something ..")
        audio = r.listen(source)
        try:
            answer = r.recognize_google(audio)
        except Exception as e:
            print(e)
    return answer

# Getting the required data
pickle_in = open("training_data.pickle","rb")
training_data = pickle.load(pickle_in)

# Model prerequisites
T = 35 # Maximum length
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(training_data)
model_path = "NLP_SARCASTIC.h5"

# Loading Down the model
# Checking top snake handler leaves sinking huckabee campaign
# Checking the Grandmother is happy
model = load_model(model_path)
sentence = []
while True:
    sentence.clear()
    he = get_string()
    sentence.append(he)
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=T)
    p = model.predict(padded)[0][0]
    p = round(p)
    if(p == 1.0):
        statement = 'Sarcastic'
    elif(p == 0.0):
        statement = 'Common Sentence'
    converter.say(statement)
    converter.runAndWait()
