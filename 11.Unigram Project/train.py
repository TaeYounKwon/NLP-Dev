from lstm import build_LSTM
from transformer import build_transformer
from pre_process import load_and_prepare
import tensorflow as tf 
import numpy as np

def get_vocab_size(f_vocab):
    with open(f_vocab, 'r', encoding='utf-8') as file:
        vocab_size = len(file.readlines())
    return vocab_size

def perplexity(labels, logits):
    loss = tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    perplexity = tf.exp(tf.reduce_mean(loss))
    return perplexity.numpy()


# parameters
train_file = './data/ptb.train.txt'
test_file = './data/ptb.test.txt'
train_dataset = load_and_prepare(train_file)
test_dataset = load_and_prepare(test_file)

train_dataset = train_dataset.filter(lambda x, y: tf.shape(x)[0] > 0 and tf.shape(y)[0] > 0)
test_dataset = test_dataset.filter(lambda x, y: tf.shape(x)[0] > 0 and tf.shape(y)[0] > 0)

vocab_size = get_vocab_size('./ptb_unigram.vocab')
embed_dim = 256
num_heads = 4
ff_dim = 1024
num_layers = 3

# Build the model
lstm_model = build_LSTM(vocab_size, embed_dim, num_layers)
# lstm_model.summary()
transformer_model = build_transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers )
# transformer_model.summary()


lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
lstm_model.fit(train_dataset, epochs=10)
exit()
# transformer not working...
transformer_model.fit(train_dataset, epochs=10)

total_perplexity = []
for batch in test_dataset:
    inputs, labels = batch
    predictions = lstm_model.predict(inputs)
    batch_perplexity = perplexity(labels, predictions)
    total_perplexity.append(batch_perplexity)

# Calculate average perplexity across all batches
average_perplexity = sum(total_perplexity) / len(total_perplexity)
print(f"Average Perplexity: {average_perplexity}")

total_perplexity = []
for batch in test_dataset:
    inputs, labels = batch
    transformer_perplexity = perplexity(labels, transformer_model.predict(test_dataset))
    batch_perplexity = perplexity(labels, predictions)
    total_perplexity.append(batch_perplexity)

# Calculate average perplexity across all batches
average_perplexity = sum(total_perplexity) / len(total_perplexity)
print(f"Average Perplexity: {average_perplexity}")

# # Assuming you have a 'test_dataset' loaded similarly
# lstm_perplexity = perplexity(test_labels, lstm_model.predict(test_dataset))
# transformer_perplexity = perplexity(test_labels, transformer_model.predict(test_dataset))

# print("LSTM Perplexity:", lstm_perplexity)
# print("Transformer Perplexity:", transformer_perplexity)