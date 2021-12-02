import numpy as np
import json
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('intents.json') as file:
    data = json.load(file)


def get_data(data):

    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


    return training_sentences, training_labels, labels, responses


def encode_labels(training_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(training_labels)
    training_labels = label_encoder.transform(training_labels)
    return training_labels


def tokenize_labels(training_sentences, vocab_size, max_len, oov_token):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    return padded_sequences, word_index


def create_model(vocab_size, embedding_dim, max_len, num_classes):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    return model


def train_model(num_epochs, model, X_train, y_train):
    history = model.fit(X_train, np.array(y_train), epochs=num_epochs)
    return history



training_sentences, training_labels, labels, responses = get_data(data)

num_classes = len(labels)
vocab_size = 1000
max_len = 20
embedding_dim = 16

padded_sequences, word_index = tokenize_labels(training_sentences, vocab_size, max_len, '<OOV>')
model = create_model(vocab_size, embedding_dim,max_len, num_classes)
training_labels = encode_labels(training_labels)
history = train_model(100, model, padded_sequences, training_labels)
