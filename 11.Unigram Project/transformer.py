import tensorflow as tf
from keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, Input, TimeDistributed
from keras.models import Model

# def transformer_decoder_block(dec_input, enc_output, embed_dim, num_heads, ff_dim, rate=0.1):
    # Self-attention with look-ahead mask (and padding mask)
    # Note: Masking is assumed to be done externally if necessary and passed to this function
    # self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    # attn_out1 = self_attention(dec_input, dec_input)  # Self-attention; mask should be included in call if necessary
    # attn_out1 = Dropout(rate)(attn_out1)
    # out1 = LayerNormalization(epsilon=1e-6)(dec_input + attn_out1)

    # # Cross-attention with encoder outputs and padding mask
    # cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    # attn_out2 = cross_attention(out1, enc_output, enc_output)  # Cross-attention; mask should be included in call if necessary
    # attn_out2 = Dropout(rate)(attn_out2)
    # out2 = LayerNormalization(epsilon=1e-6)(out1 + attn_out2)

    # # Feed-forward network
    # ffn = tf.keras.Sequential([
    #     Dense(ff_dim, activation='relu'),  # (batch_size, seq_len, ff_dim)
    #     Dense(embed_dim)                   # (batch_size, seq_len, embed_dim)
    # ])
    # ffn_output = ffn(out2)
    # ffn_output = Dropout(rate)(ffn_output)
    # sequence_output = LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

    # return sequence_output

def build_transformer(vocab_size, d_model=256, num_layers=3, num_heads=4):
    # inputs = Input(shape=(None,))
    # enc_outputs = Input(shape=(None, embed_dim))

    # embeddings = Embedding(vocab_size, embed_dim)(inputs)

    # x = embeddings
    # for _ in range(num_layers):
    #     x = transformer_decoder_block(x, enc_outputs, embed_dim, num_heads, ff_dim, rate)

    # outputs = Dense(vocab_size, activation='softmax')(x)  # Output layer
    # return Model(inputs=[inputs, enc_outputs], outputs=outputs)
    input_layer = Input(shape=(None,))
    x = Embedding(vocab_size, d_model)(input_layer)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attn_output)
        x = Dropout(0.1)(x)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)