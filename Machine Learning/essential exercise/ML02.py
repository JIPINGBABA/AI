from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



def knn_iris():
    """
    use KNN algorithm to classify iris
    :return: 
    """
    #1. get dataset
    iris=load_iris()

    #2.Splitting the dataset
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=6)

    #3.feature engineering,standardization
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.KNN algorithm estimator
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    #5 model evaluation
    #method1
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("Compare the actual and predicted values:\n",y_test==y_predict)
    #method2 Calculate accuracy
    score=estimator.score(x_test,y_test)
    print("accuracy:\n",score)
    return None


def knn_iris_gscv():
    """
    use KNN algorithm to classify iris with grid search and cross validation
    :return:
    """
    # 1. get dataset
    iris = load_iris()

    # 2.Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=6)

    # 3.feature engineering,standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN algorithm estimator
    estimator = KNeighborsClassifier(n_neighbors=3)
    #add grid search and cross validation
    #prepare paramters
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train, y_train)

    # 5 model evaluation
    # method1
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("Compare the actual and predicted values:\n", y_test == y_predict)
    # method2 Calculate accuracy
    score = estimator.score(x_test, y_test)
    print("accuracy:\n", score)

    # #best_params
    print("best_params:\n",estimator.best_params_)
    # best_score
    print("best_score:\n",estimator.best_score_)
    #best_estimator
    print("best_estimator:\n",estimator.best_estimator_)
    #cv_results
    print("cv_result:\n",estimator.cv_results_)
    return None



if __name__ == '__main__':
    #1.use KNN algorithm to classify iris
    # knn_iris()
    #2.use KNN algorithm to classify iris with grid search and cross validation
    knn_iris_gscv()