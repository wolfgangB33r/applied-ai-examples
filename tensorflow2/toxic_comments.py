import io
import os
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Helper function that is used to split a large dataframe into two dataframes
def split(df, test_size=0.2):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    val = df[~msk]
    return train, val

# Load the labeled training data
comments = pd.read_csv('comments.csv.zip')

# First start with filling non-existing comments with the empty string
comments['comment_text'].fillna(value='', inplace=True)

# Split into train and validation dataframes
train, val = split(comments)

# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    # convert to lower case
    lowercase = tf.strings.lower(input_data)
    # strip text from new lines
    s1 = tf.strings.regex_replace(lowercase, '\n', ' ')
    return tf.strings.regex_replace(s1, '[%s]' % re.escape(string.punctuation), '')

train_texts = tf.constant((comments['comment_text']))
target = tf.constant((comments['toxic']))

# The text vectorization layer is used to normalize, split, and map strings to
# integers. The layer uses the custom standardization function defined above to
# clean the texts.
# The sequence length normalizes all sentences to the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(train_texts)

embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    x=train_texts, y=target,
    epochs=1)

# Safe toxic comment word embedding to disk
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('comments_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('comments_metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()


print(model.predict_classes(tf.constant(np.array([
    'never mind it is not important', 
    'i dont care at all, fuck you', 
    'i love you']))))