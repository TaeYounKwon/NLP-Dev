import tensorflow as tf 
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, MultiHeadAttention, Dropout, LayerNormalization
from keras.models import Sequential

def build_LSTM(vocab_size, d_model = 256, num_layers=3):
    model = Sequential()
    model.add(Embedding(vocab_size, d_model))
    for _ in range(num_layers):
        model.add(LSTM(d_model, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model

def build_Transformer(vocab_size, d_model=256, num_layers=3, num_heads=4):
    input_layer = Input(shape=(None,))
    x = Embedding(vocab_size, d_model)(input_layer)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attn_output)
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)
 
lstm_model = build_LSTM(1024)
transformer_model = build_Transformer(1024)

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')