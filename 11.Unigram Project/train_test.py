from lstm_transformer import lstm_model, transformer_model
from data_pipeline import train_dataset_LSTM, test_dataset_LSTM, valid_dataset_LSTM, train_dataset_Trans, test_dataset_Trans, valid_dataset_Trans
import tensorflow as tf 

def train_and_evaluate(model, train_data, test_data, valid_data):
    model.fit(train_data, epochs=10,  validation_data =valid_data)
    loss = model.evaluate(test_data)
    perplexity = tf.exp(loss)
    print(f"Perplexity: {perplexity}")

train_and_evaluate(transformer_model, train_dataset_Trans, test_dataset_Trans,valid_dataset_Trans)
train_and_evaluate(lstm_model, train_dataset_LSTM, test_dataset_LSTM,valid_dataset_LSTM)
