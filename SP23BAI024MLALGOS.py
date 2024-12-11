import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score ,classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest


# A method for displaying confusion matrix and classification report
def report(x_test, y_test, model):
    y_pred = model.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    displayLabel = ["Non-Fraud", "Fraud"]
    display = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels= displayLabel)
    display.plot()
    plt.show()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# A method for displaying bar chart
def barChart(x_test, y_test, model):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [accuracy, precision, recall, f1, roc_auc]

    plt.figure(figsize=(6,5))
    plt.bar(metrics,values,color=['red','yellow','blue','green','pink'])
    plt.title('Evaluation Metrics')
    plt.show()



# Reading from csv file
creditCard = pd.read_csv('creditcard.csv')
x = creditCard.iloc[:,:-1]
y = creditCard['Class']

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# Training LightGBM 
lgbm = LGBMClassifier(
    objective = 'binary',
    boosting_type =  "gbdt",
    force_col_wise =True,
    random_state = 42
    )

lgbm.fit(x_train, y_train)

lgbmYpred = lgbm.predict(x_test)
LgbmAccuracy = accuracy_score(y_test,lgbmYpred)
print("Accuracy with default parameters using LightGBM: ",LgbmAccuracy)

report(x_test,y_test, lgbm)
barChart(x_test, y_test,lgbm)


# Training svm
svm = SVC(
    random_state = 42,
    probability=True
    )

svm.fit(x_train,y_train)
SvmAccuracy = svm.score(x_test,y_test)
print("Accuracy with default parameters using SVM: ",SvmAccuracy)

report(x_test,y_test, svm)
barChart(x_test, y_test, svm)


# Handling missing values
if x_train.isnull().sum().sum() > 0 or x_test.isnull().sum().sum() > 0:
    missingVal = SimpleImputer()
    x_train = pd.DataFrame(missingVal.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(missingVal.transform(x_test), columns=x_test.columns)

if y_train.isnull().sum().sum() > 0 or y_test.isnull().sum().sum() > 0:
    x_train = x_train[y_train.notnull()]
    y_train = y_train.dropna()
    x_test = x_test[y_test.notnull()]
    y_test = y_test.dropna()


# Feature selection
mi = mutual_info_classif(x_train, y_train)
mi_df = pd.DataFrame(mi, index = x_train.columns, columns = ['Mutual Information'])
mi_df.sort_values(by = 'Mutual Information', ascending = False)
selectedColumns = SelectKBest(mutual_info_classif, k = 12)
selectedColumns.fit(x_train, y_train)
x_train_selected = selectedColumns.transform(x_train)
x_test_selected = selectedColumns.transform(x_test)


# Balancing the data
smote = SMOTE(random_state=42)
smX, smY = smote.fit_resample(x_train_selected,y_train)

# Scaling the data
scaler = StandardScaler()
xScale_train = scaler.fit_transform(smX)
xScale_test = scaler.transform(x_test_selected)


# Training Lightgbm after preprocessing
lgbmScaled = LGBMClassifier(
    objective = 'binary',
    boosting_type =  "gbdt",
    force_col_wise =True,
    random_state = 42
    )


lgbmScaled.fit(xScale_train, smY)

yLgbmScaled_pred = lgbmScaled.predict(xScale_test)
LgbmScaledAccuracy = accuracy_score(y_test,yLgbmScaled_pred)
print("Accuracy with default parameters using LightGBM after scaling and all that: ",LgbmScaledAccuracy)


# Training svm after Preprocessing
svmScaled = SVC(
    random_state = 42
    )

svmScaled.fit(xScale_train,smY)
SvmScaledAccuracy = svmScaled.score(xScale_test,y_test)
print("Accuracy with default parameters using SVM after scaling and all that: ",SvmScaledAccuracy)


# Hyperparameters for lightGbm tuning
parameterGridLGBM  = {
    "num_leaves": [31, 50, 100],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "min_data_in_leaf": [10, 20, 30],
    "feature_fraction": [0.6, 0.8, 0.9]
}

# Tuning LightGbm using Grid Search
lgbmGrid = LGBMClassifier(
    objective = 'binary',
    boosting_type =  "gbdt",
    force_col_wise =True,
    random_state = 42
)
gridLgbm = GridSearchCV(estimator= lgbmGrid, param_grid= parameterGridLGBM, scoring='accuracy', cv = 3)
gridLgbm.fit(xScale_train, smY)
print("Best score: ",gridLgbm.best_score_)
print("best parameters: ", gridLgbm.best_params_)

report(xScale_test,y_test, gridLgbm)
barChart(xScale_test, y_test, gridLgbm)


# Tuning lightgbm using Random Search
lgbmRandom = LGBMClassifier(
    objective = 'binary',
    boosting_type =  "gbdt",
    force_col_wise =True,
    random_state = 42
)
randomLgbm = RandomizedSearchCV(estimator=lgbmRandom, param_distributions= parameterGridLGBM, n_iter=10, scoring='accuracy', cv = 5)
randomLgbm.fit(xScale_train, smY)
print("Best score: ",randomLgbm.best_score_)
print("best parameters: ", randomLgbm.best_params_)

report(xScale_test,y_test, randomLgbm.best_estimator_)
barChart(xScale_test, y_test, randomLgbm.best_estimator_)



# Hyperparameters for Svm tuning
parameterGridSVM = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}


# Tuning svm using Grid search
svmGrid = SVC(
    random_state = 42,
    probability=True
    )
gridsvm = GridSearchCV(estimator= svmGrid, param_grid= parameterGridSVM, scoring='accuracy', cv = 3)
gridsvm.fit(xScale_train, smY)
print("Best score: ",gridsvm.best_score_)
print("best parameters: ", gridsvm.best_params_)

report(xScale_test,y_test, gridsvm.best_estimator_)
barChart(xScale_test, y_test, gridsvm.best_estimator_)



# Tuning svm using Random Search
svmRandom = SVC(
    random_state = 42,
    probability=True
)
randomsvm = RandomizedSearchCV(estimator=svmRandom, param_distributions= parameterGridSVM, n_iter=10, scoring='accuracy', cv = 3)
randomsvm.fit(xScale_train, smY)
print("Best score: ",randomsvm.best_score_)
print("best parameters: ", randomsvm.best_params_)

report(xScale_test,y_test, randomsvm.best_estimator_)
barChart(xScale_test, y_test, randomsvm.best_estimator_)