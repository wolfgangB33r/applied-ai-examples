import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Helper function that is used to split a large dataframe into two dataframes
def split(df, test_size=0.2):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    val = df[~msk]
    return train, val

# Load Titanic dataset.
url = 'http://storage.googleapis.com/tf-datasets'
train = pd.read_csv(url + '/titanic/train.csv')
# Select the target feature
train = train.rename(columns={'survived': 'target'})
# split training set into train and eval sets
train, val = split(train, 0.2)
test = pd.read_csv(url + '/titanic/eval.csv')
print(train.head())

# Describe the feature columns of the input data set
feature_columns = []

# numeric cols
for header in ['age', 'n_siblings_spouses', 'parch', 'fare']:
  feature_columns.append(feature_column.numeric_column(header))

# indicator_columns
for col_name in ['sex', 'class', 'deck', 'embark_town', 'alone']:
  categorical_column = feature_column.categorical_column_with_vocabulary_list(
      col_name, test[col_name].unique())
  indicator_column = feature_column.indicator_column(categorical_column)
  feature_columns.append(indicator_column)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50)

print(model.evaluate(val_ds))

test['target'] = 0
test_ds = df_to_dataset(test, batch_size=batch_size)

print(model.predict_classes(test_ds))

