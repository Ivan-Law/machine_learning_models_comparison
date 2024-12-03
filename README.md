# Machine Learning Models Comparison
* Comparison on various ML classification models using Sci-kit Learn, TensorFlow and Pytorch.
* Comparison is based on Accuracy, Precision, Recall and F1 Score.
* Cross-validation with multiple Sklearn algorithms is also illustrated.
* Model tuning: GridSearchCV is used for finding the optimal values of hyperparameters to maximize model performance on Sklean models.

# Models for comparison:
* Linear Discriminant Analysis
* Support Vector Machines
* Support Vector Machines (GridSearchCV tuned)
* Decision Tree Classifier
* Decision Tree Classifier (GridSearchCV tuned)
* k-nearest neighbors (KNN)
* k-nearest neighbors (KNN) (GridSearchCV tuned)
* Random Forest
* Random Forest (GridSearchCV tuned)
* XGBoost
* XGBoost (GridSearchCV tuned)
* Neural Network - Using Pytorch
* Neural Network - Using TensorFlow

## Models comparison table in terms of Accuracy, Precision, Recall and F1 Score
```
                              Accuracy  Precision    Recall  F1 Score
ModelName                                                            
Linear Discriminant Analysis  0.968085   1.000000  0.914286  0.955224
Support Vector Machines       0.904255   1.000000  0.742857  0.852459
SVM (GridSearchCV tuned)      0.936170   0.967742  0.857143  0.909091
Decision Tree Classifier      0.909574   0.920635  0.828571  0.872180
DTC (GridSearchCV tuned)      0.930851   0.967213  0.842857  0.900763
KNN                           0.930851   0.983051  0.828571  0.899225
KNN (GridSearchCV tuned)      0.941489   0.983607  0.857143  0.916031
Random Forest                 0.962766   1.000000  0.900000  0.947368
RF (GridSearchCV tuned)       0.962766   1.000000  0.900000  0.947368
XGBoost                       0.968085   1.000000  0.914286  0.955224
XGB (GridSearchCV tuned)      0.973404   1.000000  0.928571  0.962963
PyTorch Neural Network        0.946809   0.983871  0.871429  0.924242
TensorFlow Neural Network     0.930851   1.000000  0.814286  0.897638
```

# Dataset is downloaded from 
* https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# EDA

* Check unused columns for data cleaning
<img src="https://github.com/user-attachments/assets/428f1c77-f620-42c8-bdfe-4fc62192bb5e" width="800" />

* Pie chart for diagnosis distribution
<img src="https://github.com/user-attachments/assets/5ef559a5-5118-4c82-aad8-40d01f0ecb54" width="300" />

* Count plot for Total Benign x Malignant Cells
<img src="https://github.com/user-attachments/assets/515b337c-a2d3-485d-bb12-a73f3579ebea" width="400" />

* Scatter plot for malignant and benign
<img src="https://github.com/user-attachments/assets/1a5f32b2-9fa4-40f3-afa5-662394b9d5a5" width="600" />

* Heatmap of correlation
<img src="https://github.com/user-attachments/assets/b9dd4a34-c03f-4850-af24-c082da0f8174" width="1000" />

* Boxplot for 4 chosen features
<img src="https://github.com/user-attachments/assets/b89c3051-0960-46f3-a6fd-bce8e1a10723" width="800" />


## Data columns (1 of 3)
``` 
         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  smoothness_mean  compactness_mean  concavity_mean  concave points_mean  symmetry_mean  fractal_di![05](https://github.com/user-attachments/assets/88893a4b-abfc-4488-951d-961a220a2662)
mension_mean  radius_se  \
0    842302         M        17.99         10.38           122.8     1001.0          0.11840           0.27760          0.3001              0.14710         0.2419                 0.07871     1.0950   
1    842517         M        20.57         17.77           132.9     1326.0          0.08474           0.07864          0.0869              0.07017         0.1812                 0.05667     0.5435   
2  84300903         M        19.69         21.25           130.0     1203.0          0.10960           0.15990          0.1974              0.12790         0.2069                 0.05999     0.7456   
```

## Data columns (2 of 3)
```
   texture_se  perimeter_se  area_se  smoothness_se  compactness_se  concavity_se  concave points_se  symmetry_se  fractal_dimension_se  radius_worst  texture_worst  perimeter_worst  area_worst  \
0      0.9053         8.589   153.40       0.006399         0.04904       0.05373            0.01587      0.03003              0.006193         25.38          17.33            184.6      2019.0   
1      0.7339         3.398    74.08       0.005225         0.01308       0.01860            0.01340      0.01389              0.003532         24.99          23.41            158.8      1956.0   
2      0.7869         4.585    94.03       0.006150         0.04006       0.03832            0.02058      0.02250              0.004571         23.57          25.53            152.5      1709.0   
```

## Data columns (3 of 3)
```
   smoothness_worst  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst  Unnamed: 32  
0            0.1622             0.6656           0.7119                0.2654          0.4601                  0.11890          NaN  
1            0.1238             0.1866           0.2416                0.1860          0.2750                  0.08902          NaN  
2            0.1444             0.4245           0.4504                0.2430          0.3613                  0.08758          NaN
```

## Data Shape
```
(569, 33)
```

## Correlation between features

```
radius_mean         perimeter_mean      0.998
radius_worst        perimeter_worst     0.994
radius_mean         area_mean           0.987
perimeter_mean      area_mean           0.987
radius_worst        area_worst          0.984
perimeter_worst     area_worst          0.978
radius_se           perimeter_se        0.973
perimeter_mean      perimeter_worst     0.970
radius_mean         radius_worst        0.970
perimeter_mean      radius_worst        0.969
radius_mean         perimeter_worst     0.965
area_mean           radius_worst        0.963
area_mean           area_worst          0.959
area_mean           perimeter_worst     0.959
radius_se           area_se             0.952
perimeter_mean      area_worst          0.942
radius_mean         area_worst          0.941
perimeter_se        area_se             0.938
concavity_mean      concave points_mean 0.921
```

## GridSearchCV tuning to search optimal parameters for various sklearn models
```
Time used 0:06:26.767738
=== Best Parameters for SVM =======================
{'C': 3.0, 'gamma': 0.02, 'kernel': 'linear'}
===================================================
Time used 0:00:00.242891
=== Best Parameters for DTC =======================
{'criterion': 'entropy', 'max_depth': 6, 'splitter': 'random'}
===================================================
Time used 0:00:02.079155
=== Best Parameters for KNN =======================
{'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 15, 'weights': 'distance'}
===================================================
=== Best Parameters for RFC =======================
Time used 0:00:12.589125
{'max_depth': 6, 'max_features': 'log2', 'n_estimators': 128}
=== Best Parameters ===============================
=== Best Parameters for XGB =======================
Time used 0:00:50.918040
{'colsample_bytree': 0.5, 'eta': 0.12, 'max_depth': 4, 'subsample': 0.5}
=== Best Parameters ===============================
```

## Running Neural Network using Pytorch and TensorFlow
```
Epoch: 0 Loss 18.64527702331543
Epoch: 100 Loss 0.20218220353126526
Epoch: 200 Loss 0.19413164258003235
Epoch: 300 Loss 0.18832571804523468
Epoch: 400 Loss 0.1829618513584137
Epoch: 500 Loss 0.17744486033916473
Epoch: 600 Loss 0.1717202067375183
Epoch: 700 Loss 0.16583466529846191
Epoch: 800 Loss 0.15982452034950256
Epoch: 900 Loss 0.1537085473537445
Epoch: 1000 Loss 0.1474948674440384
Epoch: 1100 Loss 0.14119695127010345
Epoch: 1200 Loss 0.1348385363817215
Epoch: 1300 Loss 0.12847214937210083
Epoch: 1400 Loss 0.12219925224781036
Epoch: 1500 Loss 0.11615904420614243
Epoch: 1600 Loss 0.11050570011138916
Epoch: 1700 Loss 0.10537530481815338
Epoch: 1800 Loss 0.1008419543504715
Epoch: 1900 Loss 0.09690515697002411
```

## Cross-validation with multiple Sklearn algorithms
```
 
SVC(random_state=42)
Cross val score: [0.85087719 0.89473684 0.92982456 0.94736842 0.9380531 ]
Average score: 0.9121720229777983

KNeighborsClassifier()
Cross val score: [0.88596491 0.93859649 0.93859649 0.94736842 0.92920354]
Average score: 0.9279459711224964

DecisionTreeClassifier(random_state=42)
Cross val score: [0.9122807  0.90350877 0.92982456 0.95614035 0.88495575]
Average score: 0.9173420276354604

RandomForestClassifier(random_state=42)
Cross val score: [0.92105263 0.93859649 0.98245614 0.96491228 0.97345133]
Average score: 0.9560937742586555

XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=42, ...)
Cross val score: [0.96491228 0.95614035 0.99122807 0.96491228 0.97345133]
Average score: 0.9701288619779538

LinearDiscriminantAnalysis()
Cross val score: [0.95614035 0.96491228 0.94736842 0.96491228 0.96460177]
Average score: 0.9595870206489675
```
