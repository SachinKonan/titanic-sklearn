import pandas as pd
from sklearn import preprocessing
from sklearn import tree

# Load training data

train_df = pd.read_csv('data/train.csv')

print '-- Training data'
print train_df.head(10)
print train_df.dtypes
print ''


# Data cleansing

mean_age = train_df['Age'].mean()

train_df['Age'] = train_df['Age'].fillna(mean_age)

print '-- After data cleansing'
print train_df.head(10)
print ''


# Prepare the label

labels = train_df['Survived']


# Prepare the features

sex_encoder = preprocessing.LabelBinarizer()
sex_encoder = sex_encoder.fit(train_df['Sex'])
train_df['Sex_encoded'] = sex_encoder.transform(train_df['Sex'])

features = train_df[['Age', 'Sex_encoded', 'Pclass']]


# Train the model

dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model = dt.fit(features, labels)

print '-- Model'
print model
print ''


# Load test data + data cleansing

test_df = pd.read_csv('data/test.csv')

test_df['Age'] = test_df['Age'].fillna(mean_age)

print '-- Test data after data cleansing'
print test_df.head(10)
print ''


# Make predictions

test_df['Sex_encoded'] = sex_encoder.transform(test_df['Sex'])

test_features = test_df[['Age', 'Sex_encoded', 'Pclass']]

test_df_results = model.predict(test_features)

test_df['prediction'] = test_df_results

print 'Test data with predictions'
print test_df.head(10)
print ''


# Compare predictions with expected results (raw)

expected_results_df = pd.read_csv('data/gender_submission.csv')

test_df['Survived'] = expected_results_df['Survived']

print '-- Predictions (raw)'
print test_df.head(10)
print ''


# Compare predictions with expected results (aggregates)

true_negatives  = test_df.loc[lambda df: (df.Survived == 0) & (df.prediction == 0), :].shape[0]
true_postives   = test_df.loc[lambda df: (df.Survived == 1) & (df.prediction == 1), :].shape[0]
false_negatives = test_df.loc[lambda df: (df.Survived == 1) & (df.prediction == 0), :].shape[0]
false_postives  = test_df.loc[lambda df: (df.Survived == 0) & (df.prediction == 1), :].shape[0]

print '-- Prediction (aggregates)'
print 'True negatives: '  + str(true_negatives)
print 'True positives: '  + str(true_postives)
print 'False negatives: ' + str(false_negatives)
print 'False positives: ' + str(false_postives)
