# TaeYoun Kwon
# Student ID: 2024010573
# Class: Speech Recognition 

import sentencepiece as spm
import tensorflow as tf 

sp = spm.SentencePieceProcessor()
sp.load('./ptb_unigram.model')

def train_sentencepiece(input_file, model_prefix, vocab_size=1024):
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix,
                                    vocab_size=vocab_size, model_type='unigram')
 
def tokenize_and_create_pairs_LSTM(text):
    token_ids = sp.encode_as_ids(text.numpy().decode('utf-8'))
    input_seq = tf.constant(token_ids[:-1], dtype=tf.int32)
    target_seq = tf.constant(token_ids[1:], dtype=tf.int32)
    return input_seq, target_seq

def load_and_prepare_LSTM(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_and_create_pairs_LSTM, inp=[x], Tout=(tf.int32, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.padded_batch(
        batch_size=32, 
        padded_shapes=([None], [None]),  
        padding_values=(0, 0)            
    ).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def tokenize_pairs(text):
    
    token_ids = sp.encode_as_ids(text.numpy().decode('utf-8'))
    
    return token_ids[:-1], token_ids[1:]

def prepare_transformer_dataset(file_path, sp, batch_size=32):
    dataset = tf.data.TextLineDataset(file_path)

    def tokenize_pairs(text):
        token_ids = sp.encode_as_ids(text.numpy().decode('utf-8'))
        return token_ids[:-1], token_ids[1:]  
    
    def to_tensor(inputs, targets):
        return (inputs, targets[:-1]), targets[1:]  

    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_pairs, inp=[x], Tout=(tf.int32, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
        batch_size, 
        padded_shapes=(([None], [None]), [None]), 
        padding_values=((0, 0), 0)                
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

train_sentencepiece('./data/ptb.test.txt', 'ptb_unigram') 
train_dataset_LSTM = load_and_prepare_LSTM('./data/ptb.train.txt') 
valid_dataset_LSTM = load_and_prepare_LSTM('./data/ptb.valid.txt')
test_dataset_LSTM = load_and_prepare_LSTM('./data/ptb.test.txt')

train_dataset_Trans = prepare_transformer_dataset('./data/ptb.train.txt',sp) 
valid_dataset_Trans = prepare_transformer_dataset('./data/ptb.valid.txt',sp)
test_dataset_Trans = prepare_transformer_dataset('./data/ptb.test.txt',sp)


for inputs, targets in train_dataset_LSTM.take(1):
    print(f"Encoder inputs shape: {inputs[0].shape}")  
    print(f"Decoder inputs shape: {inputs[1].shape}")  
    print(f"Targets shape: {targets.shape}")   
    
for inputs, targets in train_dataset_Trans.take(1):
    print(f"Encoder inputs shape: {inputs[0].shape}")  
    print(f"Decoder inputs shape: {inputs[1].shape}")  
    print(f"Targets shape: {targets.shape}")   