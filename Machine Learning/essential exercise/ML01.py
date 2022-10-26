from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import pandas as pd
from scipy.stats import pearsonr

def datasets_demo():
    """
    sklearn dataset
    :return:
    """
    iris=load_iris()
    print("iris database:\n",iris)
    print("description:\n",iris["DESCR"])
    print("feature_names\n",iris.feature_names)
    print("feature_values\n",iris.data,iris.data.shape)
    print("target:",iris.target)
    #database split
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("train_set feature:\n",x_train,x_train.shape)
    return None

def dict_demo():
    """
    The dictionary features
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 实例一个转换器类
    transfer = DictVectorizer()
    # 调用fit_transform（）
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名称:\n", transfer.get_feature_names())
    return None

def count_demo():
    """
    文本特征抽取： CountVecotrizer
    :return:
    """
    data = {"Life is short,i like like python", "Life is too long,i dislike python"}
    #1、实例化一个转换器类
    transfer=CountVectorizer(stop_words=["is","too"])
    #2、调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("feature name:\n",transfer.get_feature_names())
    return None
def minmax_demo():
    """
    normalization
    :return:
    """
    #1.get data
    data=pd.read_csv("dating.csv")
    data=data.iloc[:,:3]
    print("data:\n",data)
    #2.Instantiate a transformer
    transfer=MinMaxScaler()
    #3、call fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def stand_demo():
    """
       normalization
       :return:
       """
    # 1.get data
    data = pd.read_csv("dating.csv")
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2.Instantiate a transformer
    transfer = StandardScaler()
    # 3、call fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

def variance_demo():
    """
    Filter low variance features
    :return:
    """
    #1、get data
    data=pd.read_csv("factor_returns.csv")
    data=data.iloc[:, 1:-2]
    print("data:\n",data)
    #2.Instantiate a transformer
    transfer=VarianceThreshold(threshold=10)
    # 3、call fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new,data_new.shape)

    #Calculate the correlation between two variables
    r1=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("correlation coefficient:\n",r1)
    r2=pearsonr(data['revenue'],data['total_expense'])
    print("revenue and total_expense correlation coefficient:\n",r2)
    return None

def pca_demo():
    """
    PCA decomposition
    :return:
    """
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    #Instantiate a transformer
    transfer=PCA(n_components=0.95)
    #call fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

if __name__ == '__main__':
    #code1: sklearn dataset use
    # datasets_demo()
    # code2: The dictionary features
    # dict_demo()
    #code3: CountVecotrizer
    # count_demo()
    # minmax_demo()
    # stand_demo()
    # variance_demo()
    pca_demo()
