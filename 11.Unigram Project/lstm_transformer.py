import tensorflow as tf 
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, MultiHeadAttention, Dropout, LayerNormalization
from keras.models import Sequential

def build_LSTM(vocab_size, d_model, num_layers):
    model = Sequential()
    model.add(Embedding(vocab_size, d_model))
    for _ in range(num_layers):
        model.add(LSTM(d_model, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model

# 놀랍게도 이 모델이 가장 뛰어난 성능을 보임.
def simple_MultiHeadAttention(vocab_size, d_model=256, num_layers=3, num_heads=4):
    input_layer = Input(shape=(None,))
    x = Embedding(vocab_size, d_model)(input_layer)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attn_output)
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

def build_Decoder_Transformer(vocab_size, d_model, num_layers, num_heads):
    encoder_input = Input(shape=(None,), name='encoder_input')  # Change made: removed d_model from shape
    decoder_input = Input(shape=(None,), name='decoder_input')

    # Applying embedding to both inputs
    encoder_embed = Embedding(vocab_size, d_model)(encoder_input)
    decoder_embed = Embedding(vocab_size, d_model)(decoder_input)

    x = decoder_embed
    for _ in range(num_layers):
        x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, encoder_embed, encoder_embed)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output_layer)

 
lstm_model = build_LSTM(vocab_size=1024, d_model=256, num_layers=3)
transformer_model = build_Decoder_Transformer(vocab_size=1024, d_model=256, num_layers=3, num_heads=4)

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print("LSTM Model Summary:")
lstm_model.summary()
print("Transformer Model Summary:")
transformer_model.summary()