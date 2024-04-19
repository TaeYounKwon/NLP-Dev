import tensorflow as tf 
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Input, MultiHeadAttention, Dropout, LayerNormalization
from keras.models import Sequential
from data_pipeline import train_dataset_LSTM, test_dataset_LSTM, valid_dataset_LSTM, train_dataset_Trans, test_dataset_Trans, valid_dataset_Trans

# LSTM 모델
def build_LSTM(vocab_size, d_model, num_layers):
    model = Sequential()
    model.add(Embedding(vocab_size, d_model))
    for _ in range(num_layers):
        model.add(LSTM(d_model, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model

# 멀티헤드어텐션 모델 - 놀랍게도 이 모델이 가장 뛰어난 성능을 보임.
def simple_MultiHeadAttention(vocab_size, d_model=256, num_layers=3, num_heads=4):
    input_layer = Input(shape=(None,))
    x = Embedding(vocab_size, d_model)(input_layer)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attn_output)
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

# 트랜스포머(디코더구조) 모델 
def build_Decoder_Transformer(vocab_size, d_model, num_layers, num_heads):
    # 인코더 디코더 입력값
    encoder_input = Input(shape=(None,), name='encoder_input')  
    decoder_input = Input(shape=(None,), name='decoder_input')

    # 인코더 디코더 임베딩값
    encoder_embed = Embedding(vocab_size, d_model)(encoder_input)
    decoder_embed = Embedding(vocab_size, d_model)(decoder_input)

    # HW의 모델처럼 디코더구조의 트랜스포머 모델 구현
    x = decoder_embed
    for _ in range(num_layers):
        attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x, x)
        x = LayerNormalization()(x + attn1) 
        x = Dropout(0.1)(x)

        attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, encoder_embed, encoder_embed)
        x = LayerNormalization()(x + attn2)  
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output_layer)

# 모델 트레이닝
def train_and_evaluate(model, train_data, test_data, valid_data):
    model.fit(train_data, epochs=10,  validation_data =valid_data)
    loss = model.evaluate(test_data)
    perplexity = tf.exp(loss)
    print(f"Perplexity: {perplexity}")

lstm_model = build_LSTM(vocab_size=1024, d_model=256, num_layers=3)
transformer_model = build_Decoder_Transformer(vocab_size=1024, d_model=256, num_layers=3, num_heads=4)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 모델 구조 확인
print("LSTM Model Summary:")
lstm_model.summary()
print("Transformer Model Summary:")
transformer_model.summary()

# 모델 트레이닝 시작
train_and_evaluate(transformer_model, train_dataset_Trans, test_dataset_Trans,valid_dataset_Trans)
train_and_evaluate(lstm_model, train_dataset_LSTM, test_dataset_LSTM,valid_dataset_LSTM)






