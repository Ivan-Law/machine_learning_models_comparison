import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40, 'display.width', 200)
sns.set_theme()

### Dataset:
### https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

df = pd.read_csv("data\\Cancer_Data.csv")
print(df.head(3))
print(df.shape)
print(df['id'].nunique())
print(df.columns)

### Visualize missing data
msno.matrix(df, figsize=(16, 9), fontsize=10)
plt.show()

### Data cleaning
print(df['Unnamed: 32'])
print(df['Unnamed: 32'].unique())
print(df['id'].nunique())
df = df.drop(["id", "Unnamed: 32"], axis='columns')

print('==============================')
print(df.shape)
print(df['diagnosis'].tail(3))
print(df['diagnosis'].value_counts())
print('==============================')

### Encode 'diagnosis'
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

### Pie chart for diagnosis distribution using Matploylib
diagnosis = df['diagnosis'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(diagnosis,
        autopct='%1.2f%%',
        startangle=90)
plt.legend(["Benign", "Malignant"])
plt.title("Benign x Malignant")
plt.show()

### Count plot for Total Benign x Malignant Cells using Seaborn

diagnosis = df['diagnosis'].value_counts()
ax = sns.barplot(diagnosis, palette='tab10')
for j in df['diagnosis'].unique():
    ax.bar_label(ax.containers[j])
plt.title("Total Benign x Malignant Cells")
plt.show()

### Scatter plot for malignant and benign

sns.scatterplot(data=df, x="radius_mean", y="texture_mean",
                hue="diagnosis")
plt.legend(["Malignant", "Benign"])
plt.show()

### Group by diagnosis to calculate radius and texture mean

print(df.groupby('diagnosis')[['radius_mean', 'texture_mean']].mean())

### Heatmap of correlation
fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(df.corr(), cbar=True, annot=True, ax=ax,
            cmap='coolwarm', annot_kws={"size":8},
            mask=np.triu(np.ones_like(df.corr(), dtype=bool)))
ax.set_xticklabels(df.columns, rotation=45)
plt.tight_layout()
plt.show()

### Sorted correlation with diagnosis

print(df.corrwith(df.diagnosis).sort_values(ascending=False))

### Boxplot for 4 features

features_to_plot = ['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst']
plt.figure(figsize=(14, 6))
labels = {0: 'Benign', 1: 'Malignant'}
for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 4, i+1)
    sns.boxplot(x=df.diagnosis.map(labels), y=feature, data=df, palette='Set2')
    plt.title(f"{feature} by Diagnosis")
plt.tight_layout()
plt.show()

### Show highly correlated features (correlation > 0.92)
corr_matrix = df.corr()
c = np.where(np.abs(corr_matrix) > 0.92)
correlated_features = [(corr_matrix.iloc[x, y], x, y) for x, y in zip(*c) if x != y and x < y] # avoid duplication
s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))

for v, i, j in s_corr_list:
    print(f"{corr_matrix.index[i]:<20}"
          f"{corr_matrix.columns[j]:<20}"
          f"{v:.3f}")



from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from datetime import datetime

## Split the dataset for Sklearn ########################################
encoder = preprocessing.LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

y = df['diagnosis']
X = df.drop("diagnosis", axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

### Linear Discriminant Analysis using Scikit-Learn ###################################################

classifier = LinearDiscriminantAnalysis()
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)

### Create first DataFrame ############################################################################

MLCom = pd.DataFrame({'ModelName': ['Linear Discriminant Analysis'],
      'Accuracy': [accuracy_score(y_val, y_pred)],
      'Precision': [precision_score(y_val, y_pred)],
      'Recall': [recall_score(y_val, y_pred)],
      'F1 Score': [f1_score(y_val, y_pred)]})
MLCom.set_index('ModelName', inplace=True)

### Support Vector Machines using Scikit-Learn ########################################################

classifier = SVC(probability=True, random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['Support Vector Machines'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Hyperparameter Tuning
param_grid = {
    'C': [0.5, 1.0, 2.0, 3.0],          # penalty parameter C of the error term
    'kernel': ['linear', 'rbf'],        # specifies the kernel type to be used in the algorithm
    'gamma': [0.02, 0.08, 0.1]          # kernel coefficient for 'rbf'
}

### {'C': 2.0, 'gamma': 0.02, 'kernel': 'linear'}
starttime = datetime.now()
CV_svc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_svc.fit(X_train, y_train)
print("Time used", datetime.now() - starttime)
print('=== Best Parameters for SVM =======================')
print(CV_svc.best_params_)
print('===================================================')

classifier = SVC(C=2.0, gamma=0.02, kernel='linear', random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['SVM (GridSearchCV tuned)'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Decision Tree using Scikit-Learn ##################################################################

classifier = DecisionTreeClassifier(random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['Decision Tree Classifier'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6]
}
### {'criterion': 'gini', 'max_depth': 4, 'splitter': 'random'}
starttime = datetime.now()
CV_dtc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_dtc.fit(X_train, y_train)
print("Time used", datetime.now() - starttime)
print('=== Best Parameters for DTC =======================')
print(CV_dtc.best_params_)
print('===================================================')

classifier = DecisionTreeClassifier(criterion='gini',
                                    splitter='random',
                                    max_depth=4,
                                    random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['DTC (GridSearchCV tuned)'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### KNN using Scikit-Learn ###########################################################################

classifier = KNeighborsClassifier(weights='distance')
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['KNN'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Hyperparameter Tuning
param_grid = {
    'n_neighbors': [15, 16, 17],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}
### {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
starttime = datetime.now()
CV_knn = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_knn.fit(X_train, y_train)
print("Time used", datetime.now() - starttime)
print('=== Best Parameters for KNN =======================')
print(CV_knn.best_params_)
print('===================================================')

classifier = KNeighborsClassifier(algorithm='auto',
                                  metric='manhattan',
                                  n_neighbors=15,
                                  weights='distance')
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['KNN (GridSearchCV tuned)'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]


### Random Forest using Scikit-Learn ##################################################################

classifier = RandomForestClassifier(random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['Random Forest'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                              recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Hyperparameter Tuning
param_grid = {
    'n_estimators': [120, 128, 135],      ### number of decision trees in the forest
    'max_features': ['sqrt', 'log2'],     ### number of features considered by each tree
    'max_depth': [5, 6, 7]
}
starttime = datetime.now()
CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)
print('=== Best Parameters for RFC =======================')
print("Time used", datetime.now() - starttime)
print(CV_rfc.best_params_)
print('=== Best Parameters ===============================')

classifier = RandomForestClassifier(max_features='log2',
                                    n_estimators=128,
                                    max_depth=6,
                                    random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['RF (GridSearchCV tuned)'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                              recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### XGBoost using Scikit-Learn ###########################################################################

classifier = XGBClassifier(eval_metric='logloss', random_state=42)
predictor_xgb = classifier.fit(X_train, y_train)
y_pred = predictor_xgb.predict(X_val)
MLCom.loc['XGBoost'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                        recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Hyperparameter Tuning
param_grid = {
    'eta': [0.08, 0.1, 0.12],               # Analogous to the learning rate
    'subsample': [0.5, 0.6, 0.7],           # Fraction of observations to be random samples for each tree
    'colsample_bytree': [0.5, 0.6, 0.7],    # Fraction of columns to be random samples for each tree
    'max_depth': [3, 4, 5]                  # Max depth of a tree
}

# {'colsample_bytree': 0.6, 'eta': 0.08, 'max_depth': 3, 'subsample': 0.5}
starttime = datetime.now()
CV_xgb = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
CV_xgb.fit(X_train, y_train)
print('=== Best Parameters for XGB =======================')
print("Time used", datetime.now() - starttime)
print(CV_xgb.best_params_)
print('=== Best Parameters ===============================')

classifier = XGBClassifier(colsample_bytree=0.6,
                           eta=0.08,
                           max_depth=3,
                           subsample=0.5,
                           random_state=42)
predictor = classifier.fit(X_train, y_train)
y_pred = predictor.predict(X_val)
MLCom.loc['XGB (GridSearchCV tuned)'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                              recall_score(y_val, y_pred), f1_score(y_val, y_pred)]


## Machine Learning using Pytorch ####################################################################

import torch
import torch.nn as nn
import torch.optim as optim

y = df['diagnosis'].values
X = df.drop("diagnosis", axis=1).values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

model_dict = {}
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_val)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

class CancerModel_Pytorch(nn.Module):
    def __init__(self):
        super(CancerModel_Pytorch, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CancerModel_Pytorch().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch:', epoch, 'Loss', float(loss))

model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

MLCom.loc['PyTorch Neural Network'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                               recall_score(y_val, y_pred), f1_score(y_val, y_pred)]

### Machine Learning using TensorFlow ######################################################################

import tensorflow as tf

y = df['diagnosis'].values
X = df.drop("diagnosis", axis=1).values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int64)
y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.int64)

class CancerModel_TensorFlow(tf.keras.Model):
    def __init__(self):
        super(CancerModel_TensorFlow, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(2)  # Binary classification

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

model = CancerModel_TensorFlow()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

num_epochs = 2000
model.fit(X_train_tensor, y_train_tensor, epochs=num_epochs, batch_size=32, verbose=0)

y_pred_logits = model(X_val_tensor)
y_pred = tf.argmax(y_pred_logits, axis=1).numpy()

MLCom.loc['TensorFlow Neural Network'] = [accuracy_score(y_val, y_pred), precision_score(y_val, y_pred),
                               recall_score(y_val, y_pred), f1_score(y_val, y_pred)]
print(MLCom)


### Cross-validation with multiple Sklearn algorithms ##########################################################
print('\n### Cross-validation with multiple Sklearn algorithms ###\n')

algos = [
    SVC(random_state=42),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    XGBClassifier(random_state=42),
    LinearDiscriminantAnalysis()]

for algo in algos:
    print(str(algo))
    cs = cross_val_score(algo, X, y)
    print('Cross val score:', cs)
    print('Average score:', sum(cs)/len(cs))
    print()

