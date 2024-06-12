# pipeline_utilities

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pydotplus
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score

#import statsmodels.api as sm
import glob

def load_data():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('data/*.csv')

    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df])
        first_column_value_counts = combined_df.iloc[:,0].value_counts()
        first_column_shape = combined_df.shape
        print("Loading the DataFrame...")
        display(combined_df)
        display('value counts'.title(),first_column_value_counts)
        display('shape'.title(),first_column_shape)
    return combined_df


def X_y_set():
    data = load_data()
    X = data.copy().dropna().drop(columns=data.columns[0])
    print("Display X below:")
    display(X)
    y = data[data.columns[0]].values.reshape(-1,1)
    print("y")
    display(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("X_train")
    display(X_train)
    print("X_test")
    display(X_test)
    print("y_train")
    display(y_train)
    print("y_test")
    display(y_test)
    return train_test_split(X, y)

def My_X_StandardScaler(X_train, X_test):
    X_scaler = StandardScaler().fit(X_train)
    X_train_standardscaled = X_scaler.transform(X_train)
    X_test_standardscaled = X_scaler.transform(X_test)
    print("X_train_standardscaled")
    display(X_train_standardscaled)
    print("X_test_standardscaled")
    display(X_test_standardscaled)
    return X_train_standardscaled, X_test_standardscaled

def My_MinMaxScaler(X_train, X_test):
    scaler = MinMaxScaler().fit(X_train)
    X_train_MinMaxscaled = scaler.transform(X_train)
    print("MinMax X_train:")
    display(X_train_MinMaxscaled)

    X_test_MinMaxscaled = scaler.transform(X_test)
    print("MinMax X_test:")
    display(X_test_MinMaxscaled)
    return X_train_MinMaxscaled, X_test_MinMaxscaled

def r2_adj(x, y, lr):
    r2 = lr.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def My_LinearRegression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predicted_y = lr.predict(X_test)

    mse = mean_squared_error(y_test, predicted_y)
    r2 = r2_score(y_test, predicted_y)
    rmse = np.sqrt(mse)
    score = lr.score(X_test, predicted_y)
    adj_score = r2_adj(X_test, predicted_y, lr)
    cross_validation_scores = cross_val_score(LinearRegression(), X_train, y_train, scoring = "r2")
    
    print(f"mse: {mse}")
    print(f"R2: {r2}")
    print(f"root mean squared error:  {rmse}")
    print(f"score:  {score}")
    print(f"Adjusted R2:  {adj_score}")
    print(f"All scores: {cross_validation_scores}")
    print(f"Mean score: {cross_validation_scores.mean()}")
    print(f"Standard Deviation: {cross_validation_scores.std()}")
    return mse, r2, rmse, score, adj_score, cross_validation_scores

def Standard_Classification(X_train_standardscaled, X_test_standardscaled, y_train, y_test):
    Standard_LogisticRegression(X_train_standardscaled, X_test_standardscaled, y_train, y_test)
    KNeighborsClassifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test)
    Standard_DecisionTreeModel(X_train_standardscaled, X_test_standardscaled, y_train, y_test)
    Standard_RandomForestClassifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test)
    Standard_Extra_Tree_Classifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test)

def MinMax_Classification(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):
    MinMax_LogisticRegression(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test)
    KNeighborsClassifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test)
    MinMax_DecisionTreeModel(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test)
    MinMax_RandomForestClassifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test)
    MinMax_Extra_Tree_Classifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test)


def Standard_LogisticRegression(X_train_standardscaled, X_test_standardscaled, y_train, y_test):
    logistic_regression_model = LogisticRegression()
    Standard_LogisticRegression = logistic_regression_model.fit(X_train_standardscaled, y_train)
    testing_predections = Standard_LogisticRegression.predict(X_test_standardscaled)
    pred_probas = Standard_LogisticRegression.predict_proba(X_test_standardscaled)
    pred_probas_firsts = [prob[1] for prob in pred_probas]
    pred_probas_firsts[0:5]
    accuracy_score(y_test, testing_predections)

    print(f"Training Data Score: {Standard_LogisticRegression.score(X_train_standardscaled, y_train)}")
    print(f"Testing Data Score: {Standard_LogisticRegression.score(X_test_standardscaled, y_test)}")
    print(confusion_matrix(y_test, testing_predections, labels = [1,0]))
    print(classification_report(y_test, testing_predections, labels=[1,0]))
    print(balanced_accuracy_score(y_test, testing_predections))

def MinMax_LogisticRegression(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):
    LogisticRegression = LogisticRegression()
    MinMax_LogisticRegression = LogisticRegression.fit(X_train_MinMaxscaled, y_train)
    MinMax_testing_predections = LogisticRegression.predict(X_test_MinMaxscaled)
    pred_probas = MinMax_testing_predections.predict_proba(X_test_MinMaxscaled)
    pred_probas_firsts = [prob[1] for prob in pred_probas]
    pred_probas_firsts[0:5]
    accuracy_score(y_test, MinMax_testing_predections)

    print(f"Training Data Score: {MinMax_LogisticRegression.score(X_train_MinMaxscaled, y_train)}")
    print(f"Testing Data Score: {MinMax_LogisticRegression.score(X_test_MinMaxscaled, y_test)}")
    print(confusion_matrix(y_test, MinMax_testing_predections, labels = [1,0]))
    print(classification_report(y_test, MinMax_testing_predections, labels=[1,0]))
    print(balanced_accuracy_score(y_test, MinMax_testing_predections))
    display(pred_probas)

def SupportVentorMachine(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    print('Train Accuracy: %.3f' % svm.score(X_train, y_train))
    print('Test Accuracy: %.3f' % svm.score(X_test, y_test))
    testing_predictions = svm.predict(X_test)
    display(testing_predictions)
    accuracy_score(y_test, testing_predictions)
    svm_train = svm.score(X_train, y_train)
    svm_test = svm.score(X_test, y_test)
    return svm_train, svm_test

def KNeighborsClassifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test):
    train_scores = []
    test_scores = []
    for k in range(1, 10, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_standardscaled, y_train)
        train_score = knn.score(X_train_standardscaled, y_train)
        test_score = knn.score(X_test_standardscaled, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        return train_scores, test_scores
    # Plot the results
    plt.plot(range(1, 10, 2), train_scores, marker='o', label="training scores")
    plt.plot(range(1, 10, 2), test_scores, marker="x", label="testing scores")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy score")
    plt.legend()
    plt.show()

def KNeighborsClassifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):
    train_scores = []
    test_scores = []
    for k in range(1, 10, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_MinMaxscaled, y_train)
        train_score = knn.score(X_train_MinMaxscaled, y_train)
        test_score = knn.score(X_test_MinMaxscaled, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        return train_scores, test_scores
    # Plot the results
    plt.plot(range(1, 10, 2), train_scores, marker='o', label="training scores")
    plt.plot(range(1, 10, 2), test_scores, marker="x", label="testing scores")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy score")
    plt.legend()
    plt.show()

def Standard_DecisionTreeModel(X_train_standardscaled, X_test_standardscaled, y_train, y_test):
    model = tree.DecisionTreeClassifier(random_state=42)
    model = model.fit(X_train_standardscaled, y_train)
    y_predictions = model.predict(X_test_standardscaled)
    Decision_tree_accuacy_score = accuracy_score(y_test, y_predictions)
    print(f"Decision Tree Accuracy Score: {Decision_tree_accuacy_score}")
    return Decision_tree_accuacy_score

def MinMax_DecisionTreeModel(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):
    model = tree.DecisionTreeClassifier(random_state=42)
    model = model.fit(X_train_MinMaxscaled, y_train)
    y_predictions = model.predict(X_test_MinMaxscaled)
    Decision_tree_accuacy_score = accuracy_score(y_test, y_predictions)
    print(f"Decision Tree Accuracy Score: {Decision_tree_accuacy_score}")
    return Decision_tree_accuacy_score

def Standard_RandomForestClassifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test):
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_standardscaled, y_train)
    y_predictions = random_forest_classifier.predict(X_test_standardscaled)
    print(f'Training Score: {random_forest_classifier.score(X_train_standardscaled, y_train)}')
    print(f'Testing Score: {random_forest_classifier.score(X_test_standardscaled, y_test)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_predictions)}') 
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_predictions)}')
    feature_importances = random_forest_classifier.feature_importances_
    # List the top 10 most important features
    importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
    importances_sorted[:10]

    depths = range(1, 10, 2)
    scores = {'train':[], 'test':[], 'depth':[]}
    for depth in depths:
        model = RandomForestClassifier(max_depth=depth)
        model.fit(X_train_standardscaled, y_train)
        
        train_score = model.score(X_train_standardscaled, y_train)
        test_score = model.score(X_test_standardscaled, y_test)

        scores['depth'].append(depth)
        scores['train'].append(train_score)
        scores['test'].append(test_score)

        scores_df = pd.DataFrame(scores).set_index('depth')
        display(scores_df)
        scores_df.plot()

def MinMax_RandomForestClassifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_MinMaxscaled, y_train)
    y_predictions = random_forest_classifier.predict(X_test_MinMaxscaled)
    print(f'Training Score: {random_forest_classifier.score(X_train_MinMaxscaled, y_train)}')
    print(f'Testing Score: {random_forest_classifier.score(X_test_MinMaxscaled, y_test)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_predictions)}') 
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_predictions)}')
    feature_importances = model.feature_importances_
    # List the top 10 most important features
    importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
    importances_sorted[:10]

    depths = range(1, 10, 2)
    scores = {'train':[], 'test':[], 'depth':[]}
    for depth in depths:
        model = RandomForestClassifier(max_depth=depth)
        model.fit(X_train_MinMaxscaled, y_train)
        
        train_score = model.score(X_train_MinMaxscaled, y_train)
        test_score = model.score(X_test_MinMaxscaled, y_test)

        scores['depth'].append(depth)
        scores['train'].append(train_score)
        scores['test'].append(test_score)

        scores_df = pd.DataFrame(scores).set_index('depth')
        display(scores_df)  
        scores_df.plot()
    
def Standard_Extra_Tree_Classifier(X_train_standardscaled, X_test_standardscaled, y_train, y_test):      
    model = ExtraTreesClassifier(random_state=1).fit(X_train_standardscaled, y_train)
    # Evaluate the model
    print(f'Training Score: {model.score(X_train_standardscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_standardscaled, y_test)}')
    
    # Train the Gradient Boosting classifier
    model = GradientBoostingClassifier(random_state=1).fit(X_train_standardscaled, y_train)

    # Evaluate the model
    print(f'Training Score: {model.score(X_train_standardscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_standardscaled, y_test)}')

    # Train the AdaBoostClassifier
    model = AdaBoostClassifier(random_state=1).fit(X_train_standardscaled, y_train)

    # Evaluate the model
    print(f'Training Score: {model.score(X_train_standardscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_standardscaled, y_test)}')

def MinMax_Extra_Tree_Classifier(X_train_MinMaxscaled, X_test_MinMaxscaled, y_train, y_test):      
    model = ExtraTreesClassifier(random_state=1).fit(X_train_MinMaxscaled, y_train)
    # Evaluate the model
    print(f'Training Score: {model.score(X_train_MinMaxscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_MinMaxscaled, y_test)}')
    
    # Train the Gradient Boosting classifier
    model = GradientBoostingClassifier(random_state=1).fit(X_train_MinMaxscaled, y_train)

    # Evaluate the model
    print(f'Training Score: {model.score(X_train_MinMaxscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_MinMaxscaled, y_test)}')

    # Train the AdaBoostClassifier
    model = AdaBoostClassifier(random_state=1).fit(X_train_MinMaxscaled, y_train)

    # Evaluate the model
    print(f'Training Score: {model.score(X_train_MinMaxscaled, y_train)}')
    print(f'Testing Score: {model.score(X_test_MinMaxscaled, y_test)}')
