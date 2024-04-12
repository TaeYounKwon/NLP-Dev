import tensorflow as tf 
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.models import Sequential

def build_LSTM(vocab_size, d_model = 256, num_layers=3):
    model = Sequential()
    model.add(Embedding(vocab_size, d_model))
    for _ in range(num_layers):
        model.add(LSTM(d_model, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    return model
 
 