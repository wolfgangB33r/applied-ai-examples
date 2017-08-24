import os
# disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import pandas as pd
import tempfile

def input_fn(data_file, num_epochs, shuffle):
	# Specify the columns for the CSV data file
	CSV_COLUMNS = [
		"PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
	]

	data = pd.read_csv(tf.gfile.Open(data_file), names=CSV_COLUMNS, skipinitialspace=True, engine="python", skiprows=1)
	# clear data set by removing missing values (NaN)
    #data = data.dropna(how="any", axis=0)
	# clear data set by replacing missing values (NaN)
	
	data['Age'].fillna(data['Age'].median(), inplace=True)
	data['Pclass'].fillna(data['Pclass'].median(), inplace=True)
	data['Sex'].fillna('male', inplace=True)
	data['SibSp'].fillna(data['Age'].median(), inplace=True)
	data['Fare'].fillna(data['Fare'].median(), inplace=True)
	data['Embarked'].fillna('S', inplace=True)
	survived = data['Survived']
	del data['Name']
	del data['Ticket']
	del data['Cabin']
	return tf.estimator.inputs.pandas_input_fn(x=data, y=survived, batch_size=100, num_epochs=num_epochs, shuffle=shuffle, num_threads=1)
	
def input_fn_test(data_file, num_epochs, shuffle):
	# Specify the columns for the CSV data file
	CSV_COLUMNS = [
		"PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
	]
	data = pd.read_csv(tf.gfile.Open(data_file), names=CSV_COLUMNS, skipinitialspace=True, engine="python", skiprows=1)
	# clear data set by removing missing values (NaN)
	data['Age'].fillna(data['Age'].median(), inplace=True)
	data['Pclass'].fillna(data['Pclass'].median(), inplace=True)
	data['Sex'].fillna('female', inplace=True)
	data['SibSp'].fillna(data['Age'].median(), inplace=True)
	data['Fare'].fillna(data['Fare'].median(), inplace=True)
	data['Embarked'].fillna('S', inplace=True)
	del data['Name']
	del data['Ticket']
	del data['Cabin']
	return tf.estimator.inputs.pandas_input_fn(x=data, batch_size=100, num_epochs=num_epochs, shuffle=shuffle, num_threads=1), data
	
# prepare the model features

# category features
sex = tf.feature_column.categorical_column_with_vocabulary_list("Sex", ["female", "male"])
embarked = tf.feature_column.categorical_column_with_vocabulary_list("Embarked", ["C", "Q", "S"])
# numerical features
pclass = tf.feature_column.numeric_column("Pclass")
age = tf.feature_column.numeric_column("Age")
sib = tf.feature_column.numeric_column("SibSp")
fare = tf.feature_column.numeric_column("Fare")
# engineered bucket features.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 70])
fare_buckets = tf.feature_column.bucketized_column(
    fare, boundaries=[8, 13, 31])
	
# define the columns
deep_columns = [
    tf.feature_column.indicator_column(sex),
    #tf.feature_column.indicator_column(embarked),
    age_buckets,
	#pclass,
    #sib,
    fare_buckets
]

# create the deep neural network estimator
model_dir = "./logs" # model_dir is where the TensorBoard summaries go automatically
estimator = tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 50])
	
# set num_epochs to None to get infinite stream of data.
estimator.fit(input_fn=input_fn('train.csv', num_epochs=None, shuffle=False), steps=10000)

# load data file to predict
input_pred, data = input_fn_test('test.csv', num_epochs=1, shuffle=False);
# predict unknown people
predictions = estimator.predict(input_fn=input_pred)
# write prediction into a file
f = open('predictions.csv', 'w')
f.write('PassengerId,Survived\n')
for i, p in enumerate(predictions):
	print("Prediction %s: %s" % (data['PassengerId'][i], p))	
	f.write('%s,%s\n' % (data['PassengerId'][i], p))
	


