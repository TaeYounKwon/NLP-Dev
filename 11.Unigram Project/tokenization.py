import sentencepiece as spm

def train_sentencepiece(input_file, model_prefix, vocab_size=1024):
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix,
                                    vocab_size=vocab_size, model_type='unigram')

# Train the model
train_sentencepiece('./data/ptb.test.txt', 'ptb_unigram')
