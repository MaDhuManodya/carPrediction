import warnings
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
data = 'car+evaluation/car.data'
df = pd.read_csv(data, header=None)

# View dimensions of the dataset
print(df.shape)

# Preview the dataset
print(df.head())

# Define column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
print(df.head())

# Check data types
df.info()

# Frequency distribution of values in each column
for col in col_names:
    print(df[col].value_counts())

# Check the frequency distribution for the target variable 'class'
print(df['class'].value_counts())

# Check for missing values
print(df.isnull().sum())

# Define feature vector (X) and target variable (y)
X = df.drop(['class'], axis=1)
y = df['class']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# View the shape of the training and test sets
print(X_train.shape, X_test.shape)

# Check data types in X_train
print(X_train.dtypes)

# Preview X_train
print(X_train.head())

# Encode categorical variables using Ordinal Encoding
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Preview encoded training data
print(X_train.head())
print(X_test.head())

# Instantiate Random Forest classifier with default 10 trees
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Check accuracy score with 10 trees
print('Model accuracy score with 10 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# Instantiate Random Forest classifier with 100 trees
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)

# Predict on the test set
y_pred_100 = rfc_100.predict(X_test)

# Check accuracy score with 100 trees
print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred_100)))

# View feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot feature importance
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Drop 'doors' feature and retrain the model
X = df.drop(['class', 'doors'], axis=1)
y = df['class']

# Split data into training and testing sets again (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical variables again
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Instantiate and fit Random Forest classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Check accuracy score after dropping 'doors' feature
print('Model accuracy score with doors variable removed : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

# Classification Report
print(classification_report(y_test, y_pred))
