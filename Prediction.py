import pickle
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, \
    roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

def regressorpredictbuild():
    Data=pd.read_csv('DataScaled.csv')
    X=Data[['quantity tons_log','status','item type','application','thickness_log','width','country','customer','product_ref']]
    y=Data['selling_price_log']

    ohe=OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X[['item type']])
    X_ohe=ohe.fit_transform(X[['item type']]).toarray()
    ohe2 = OneHotEncoder(handle_unknown='ignore')
    ohe2.fit(X[['status']])
    X_ohe2 = ohe2.fit_transform(X[['status']]).toarray()
    X = np.concatenate((X[['quantity tons_log', 'application', 'thickness_log', 'width', 'country', 'customer','product_ref']].values, X_ohe, X_ohe2), axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    dtr = DecisionTreeRegressor()
    param_grid = {'max_depth': [2, 5, 10, 20],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4],
                  'max_features': ['auto', 'sqrt', 'log2']}

    grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Mean squared error:', mse)
    print('R-squared:', r2)

    newtestsample = np.array([[np.log(40), 10, np.log(250), 0, 28, 30202938, 1670798778, 'PL', 'Won']])
    newtestsampleohe = ohe.transform(newtestsample[:, [7]]).toarray()
    newtestsampleohe2 = ohe2.transform(newtestsample[:, [8]]).toarray()
    newtestsample = np.concatenate((newtestsample[:, [0, 1, 2, 3, 4, 5, 6, ]], newtestsampleohe, newtestsampleohe2), axis=1)
    newtestsample1 = scaler.transform(newtestsample)
    new_pred = best_model.predict(newtestsample1)
    print('Predicted selling price:', np.exp(new_pred))

    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('t.pkl', 'wb') as f:
        pickle.dump(ohe, f)
    with open('s.pkl', 'wb') as f:
        pickle.dump(ohe2, f)

def classificationpredictbuild():
    Data = pd.read_csv('DataScaled.csv')
    data = Data[Data['status'].isin(['Won', 'Lost'])]

    Y = data['status']
    X = data[['quantity tons_log', 'selling_price_log', 'item type', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']]

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X[['item type']])
    X_ohe = ohe.fit_transform(X[['item type']]).toarray()
    lb = LabelBinarizer()
    lb.fit(Y)
    y = lb.fit_transform(Y)

    X = np.concatenate((X[['quantity tons_log', 'selling_price_log', 'application', 'thickness_log', 'width', 'country',
                           'customer', 'product_ref']].values, X_ohe), axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt.show())

    newtestsample = np.array([[np.log(890), np.log(558), 10, np.log(2), 1800, 28.0, 30202956, 1671789778, 'W']])
    newtestsampleohe = ohe.transform(newtestsample[:, [8]]).toarray()
    newtestsample = np.concatenate((newtestsample[:, [0, 1, 2, 3, 4, 5, 6, 7]], newtestsampleohe), axis=1)
    newtestsample = scaler.transform(newtestsample)
    new_pred = dtc.predict(newtestsample)
    if new_pred == 1:
        print('The status is: Won')
    else:
        print('The status is: Lost')

    with open('cmodel.pkl', 'wb') as file:
        pickle.dump(dtc, file)
    with open('cscaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('ct.pkl', 'wb') as f:
        pickle.dump(ohe, f)





