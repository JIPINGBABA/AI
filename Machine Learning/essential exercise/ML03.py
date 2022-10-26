from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
import joblib


def linearRegression_normalEquations():
    """
    normal equation to predict housing price in Boston
    :return:
    """
    #1.get dataset
    boston=load_boston()

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    #3.StandardScaler
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.estimator
    estimator=LinearRegression()
    estimator.fit(x_train,y_train)

    #5.concluded that the model
    print("normalEquations coef:\n",estimator.coef_)
    print("normalEquations intercept:\n",estimator.intercept_)
    #6.model evaluation
    y_predict = estimator.predict(x_test)
    print("Predict prices：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("normalEquations MSE:\n ", error)
    return None


def linearRegression_stochasticGradientDescent():
    """
    stochastic gradient descent to predict housing price in Boston
    :return:
    """
    #1.get dataset
    boston=load_boston()
    print("feature numbers:\n",boston.data.shape)

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    #3.StandardScaler
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.estimator
    estimator=SGDRegressor(learning_rate="constant",eta0=0.001,max_iter=10000)
    estimator.fit(x_train,y_train)

    #5.concluded that the model
    print("stochasticGradientDescent coef:\n",estimator.coef_)
    print("stochasticGradientDescent intercept:\n",estimator.intercept_)
    #6.model evaluation
    y_predict=estimator.predict(x_test)
    print("Predict prices：\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print("stochasticGradientDescent MSE:\n ",error)
    return None

def linearRegression_ridge():
    """
    Ridge to predict housing price in Boston
    :return:
    """
    #1.get dataset
    boston=load_boston()
    print("feature numbers:\n",boston.data.shape)

    #2.split dataset
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    #3.StandardScaler
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4.estimator
    estimator=Ridge(alpha=10)
    estimator.fit(x_train,y_train)

    #dump
    joblib.dump(estimator,"my_ridge.pkl")
    #load model
    # estimator=joblib.load("my_ridge.pkl")


    #5.concluded that the model
    print("Ridget coef:\n",estimator.coef_)
    print("Ridge intercept:\n",estimator.intercept_)
    #6.model evaluation
    y_predict=estimator.predict(x_test)
    print("Predict prices：\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print("Ridge MSE:\n ",error)
    return None
if __name__ == '__main__':
   #1  normal equation to predict housing price in Boston
   # linearRegression_normalEquations()
   # print("============================================")
   # 2 stochastic gradient descent to predict housing price in Boston
   # linearRegression_stochasticGradientDescent()
   # print("============================================")
   #3 Ridge to predict housing price in Boston
   linearRegression_ridge()