# TaeYoun Kwon
# Student ID: 2024010573
# Class: Speech Recognition 

import sentencepiece as spm
import tensorflow as tf 

sp = spm.SentencePieceProcessor()
sp.load('./ptb_unigram.model')
    
def tokenize_and_create_pairs(text):
    token_ids = sp.encode_as_ids(text.numpy().decode('utf-8'))
    input_seq = tf.constant(token_ids[:-1], dtype=tf.int32)
    target_seq = tf.constant(token_ids[1:], dtype=tf.int32)
    return input_seq, target_seq

def load_and_prepare(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_and_create_pairs, inp=[x], Tout=(tf.int32, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Padding the batches to ensure all sequences in the batch have the same length
    dataset = dataset.padded_batch(
        batch_size=32, 
        padded_shapes=([None], [None]),  # [None] indicates that the sequence length can vary
        padding_values=(0, 0)            # Pad with the token ID for <pad>, assuming it's 0
    ).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
 
train_dataset = load_and_prepare('./data/ptb.train.txt') 
test_dataset = load_and_prepare('./data/ptb.test.txt')
