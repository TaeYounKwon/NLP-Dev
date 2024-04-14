from lstm_transformer import lstm_model, transformer_model
from data_pipeline import train_dataset, test_dataset
import tensorflow as tf 

def train_and_evaluate(model, train_data, test_data):
    model.fit(train_data, epochs=10)
    loss = model.evaluate(test_data)
    perplexity = tf.exp(loss)
    print(f"Perplexity: {perplexity}")

train_and_evaluate(transformer_model, train_dataset, test_dataset)
train_and_evaluate(lstm_model, train_dataset, test_dataset)
