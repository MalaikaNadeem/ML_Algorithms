from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd


dataFrame = pd.read_csv('F:/Semester 3/DSA/DSA Lab/Lab 1/PAI/MachineLearning/Project/creditcard.csv')
print(dataFrame.head())

#https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

x = dataFrame.drop('Class', axis=1)
y = dataFrame['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


#Preprocessing

if x_train.isnull().sum().sum() > 0 or x_test.isnull().sum().sum() > 0:
    missingVal = SimpleImputer()
    x_train = pd.DataFrame(missingVal.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(missingVal.transform(x_test), columns=x_test.columns)

if y_train.isnull().sum().sum() > 0 or y_test.isnull().sum().sum() > 0:
    x_train = x_train[y_train.notnull()]
    y_train = y_train.dropna()
    x_test = x_test[y_test.notnull()]
    y_test = y_test.dropna()

mutualInfo = mutual_info_classif(x_train,y_train)
mutualInfoDf = pd.DataFrame(mutualInfo, index = x_train.columns, columns = ['Mutual Information'])
mutualInfoDf.sort_values(by = 'Mutual Information', ascending = False)
selectedColumns = SelectKBest(mutual_info_classif, k = 12)
selectedColumns.fit(x_train, y_train)
x_train_selected = selectedColumns.transform(x_train)
x_test_selected = selectedColumns.transform(x_test)

smote = SMOTE(random_state=42)
smoteX , smoteY = smote.fit_resample(x_train,y_train)

scalar = StandardScaler()
scaledX_train = scalar.fit_transform(smoteX)
scaledX_test = scalar.transform(x_test)
print(scaledX_train)


#Evaluation Metrics

def report(x_test, y_test, model):
    y_pred = model.predict(x_test)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    displayLabel = ["Non-Fraud", "Fraud"]
    display = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels= displayLabel)
    display.plot()
    plt.show()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def barChart(x_test, y_test, model):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('Accuracy', accuracy)
    precision = precision_score(y_test, y_pred)
    print('Precision' , precision)
    recall = recall_score(y_test,y_pred)
    print('Recall', recall)
    f1 = f1_score(y_test,y_pred)
    print('F1 score', f1)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print('ROC AUC Score', roc_auc)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [accuracy, precision, recall, f1, roc_auc]

    plt.figure(figsize=(9,7))
    plt.bar(metrics,values,color=['red','yellow','blue','green','pink'])
    plt.title('Evaluation Metrics')
    plt.show()


#XGBoost Algorithm

xgb_Classifier = XGBClassifier(
            booster = 'gbtree',
            objective= 'binary:logistic',
            random_state = 42
            )

xgb_Classifier.fit(x_train, y_train)
y_pred_xgb = xgb_Classifier.predict(x_test)
accuracyXGB = accuracy_score(y_test,y_pred_xgb)
print(f'Accuracy of XgB Classifier before Preprocessing : {accuracyXGB}')

xgb_Classifier.fit(scaledX_train, smoteY)
y_pred_xgb_ = xgb_Classifier.predict(scaledX_test)
accuracyXGB_ = accuracy_score(y_test,y_pred_xgb_)
print(f'Accuracy of XgB Classifier After Preprocessing : {accuracyXGB_}')


#Optimization using Random Search , Grid Search

ParametersXgB = {
        'max_depth' : [4,6,8],
        'learning_rate' : [0.1,0.3,0.5],
        'n_estimators' : [200,300,500],
        'subsample' : [0.4,0.6,1.0],
        'colsample_bytree' : [0.4,0.6,1.0],
        'scale_pos_weight' : [1, 20, 50]
}

RandomSearchXGB = RandomizedSearchCV(estimator= xgb_Classifier, param_distributions=ParametersXgB ,scoring='accuracy', n_iter=10, cv=5)
RandomSearchXGB.fit(scaledX_train,smoteY)
print(RandomSearchXGB.best_params_)
print(RandomSearchXGB.best_score_)

report(scaledX_test,y_test , RandomSearchXGB.best_estimator_)
barChart(scaledX_test,y_test , RandomSearchXGB.best_estimator_)


GridSearchXGB = GridSearchCV(estimator=xgb_Classifier , param_grid=ParametersXgB, scoring='accuracy' , cv=3)
GridSearchXGB.fit(scaledX_train,smoteY)
print(GridSearchXGB.best_params_)
print(GridSearchXGB.best_score_)

report(scaledX_test,y_test , GridSearchXGB.best_estimator_)
barChart(scaledX_test,y_test , GridSearchXGB.best_estimator_)



#Random Forest Algorithm

randomForest = RandomForestClassifier(random_state=42)

randomForest.fit(x_train, y_train)
y_pred_RF = randomForest.predict(x_test)
accuracyRF = accuracy_score(y_test,y_pred_RF)
print(f'Accuracy of RandomForest Classifier Before Preprocessing : {accuracyRF}')

randomForest.fit(scaledX_train, smoteY)
y_pred_RF_ = randomForest.predict(scaledX_test)
accuracyRF_ = accuracy_score(y_test,y_pred_RF_)
print(f'Accuracy of RandomForest Classifier After Preprocessing : {accuracyRF_}')


#Optimization using Random Search , Grid Search

ParametersRF = {
        'max_depth' : [8,10,12],
        'min_samples_split' : [4,6,8],
        'n_estimators' : [20,30,40]
}

RandomSearchRF = RandomizedSearchCV(estimator= randomForest, param_distributions=ParametersRF ,scoring='accuracy', n_iter=10, cv=5)
RandomSearchRF.fit(scaledX_train,smoteY)
print(RandomSearchRF.best_params_)
print(RandomSearchRF.best_score_)

report(scaledX_test,y_test , RandomSearchRF.best_estimator_)
barChart(scaledX_test,y_test , RandomSearchRF.best_estimator_)



GridSearchRF = GridSearchCV(estimator= randomForest , param_grid=ParametersRF, scoring='accuracy' , cv=3)
GridSearchRF.fit(scaledX_train, smoteY)
print(GridSearchRF.best_params_)
print(GridSearchRF.best_score_)

report(scaledX_test,y_test , GridSearchRF.best_estimator_)
barChart(scaledX_test,y_test , GridSearchRF.best_estimator_)