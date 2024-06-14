import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from Configuration import Config
from keras.models import load_model
import keras_tuner as kt
from keras.callbacks import ModelCheckpoint, EarlyStopping



conf = Config()
config = conf.get_config()


def get_data(train_size=0.7, val_test_split=0.5, label_columns = [1, 2, 3, 4, 5, 6, 7], lyrics_column=[8]):
    X_path = os.path.join(config.data_folder, config.data_filename)
    y_path = os.path.join(config.data_folder, config.data_filename)
    X = pd.read_csv(X_path, usecols=lyrics_column)
    y = pd.read_csv(y_path, usecols=label_columns)
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=(1-train_size), random_state=27)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = val_test_split, random_state=4)
    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize(X_train, X_val, X_test):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.vocab_size)
    tokenizer.fit_on_texts(X_train['Lyrics'])
    X_train = tokenizer.texts_to_sequences(X_train['Lyrics'])
    X_val = tokenizer.texts_to_sequences(X_val['Lyrics'])
    X_test = tokenizer.texts_to_sequences(X_test['Lyrics'])
    return X_train, X_val, X_test, tokenizer

def get_embedding_matrix(tokenizer):
    embeddings_index = dict()
    f = open(config.embedding_filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((config.vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > config.vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix

def preprocess_data(train_size=0.7, val_test_split=0.5, label_columns = [2, 3, 4, 5, 6, 7, 8], lyrics_column=[9]):
    conf = Config()
    config = conf.get_config()
    #get data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(train_size=train_size, val_test_split=val_test_split, label_columns = label_columns, lyrics_column=lyrics_column)
    #tokenize data
    X_train, X_val, X_test, tokenizer = tokenize(X_train, X_val, X_test)
    #pad data
    X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen, padding='post')
    X_val = sequence.pad_sequences(X_val, maxlen=config.maxlen, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen, padding='post')
    #get embeddings
    embedding_matrix = get_embedding_matrix(tokenizer)
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix

from ModelBuilder import build_model

def get_model(X_train, X_val, y_train, y_val):
    best_hps = None
    if config.mode == 'scratch':
        if config.tuning:
            obj = kt.Objective('val_categorical_accuracy', direction='max')
            my_tuner = kt.BayesianOptimization(hypermodel=build_model, objective=obj, max_trials=10, seed=27, max_retries_per_trial=1, max_consecutive_failed_trials=50)
            my_tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=config.search_epochs, callbacks=[ModelCheckpoint("tuning-model-{epoch:02d}ep.keras", monitor="val_categorical_accuracy", verbose=1, save_best_only=False, mode='max'), EarlyStopping(monitor='val_loss', patience=config.early_patience, restore_best_weights=True)])
            best_hps = my_tuner.get_best_hyperparameters(num_trials=1)[0]
            model = build_model(best_hps)
        else:
            model = build_model(hp=None)

    elif config.mode == 'load':
        checkpoint_path = config.checkpoint_path
        model = load_model(checkpoint_path)
    return model, best_hps