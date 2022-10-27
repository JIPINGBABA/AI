import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow_datasets


def full_connection():
    """
    full_connection
    :return:
    """
    #1.prepare dataset
    mnist=input_data.read_data_sets("./mnist_data",one_hot=True)
    # tensflow_datasets
    x=tf.placeholder(dtype=tf.float32,shape=(None,784))
    y_true=tf.placeholder(dtype=tf.float32,shape=(None,10))

    #2.build model
    Weights=tf.Variable(initial_value=tf.random_normal(shape=[784,10]))
    bias=tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict=tf.matmul(x,Weights)+bias

    #3.build loss function
    error=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))

    #4.optimizer
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error)

    #5.accurate accuracy calculation
    equal_list=tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
    accuracy=tf.reduce_mean(tf.cast(equal_list,tf.float32))





    #initialize
    init=tf.global_variables_initializer()

    #open session
    with tf.Session() as sess:
        sess.run(init)
        image,label=mnist.train.next_batch(100)
        print("before train,loss %f" % sess.run(error,feed_dict={x:image,y_true:label}))
        #begin train
        for i in range(3000):
            _,loss,accuracy_value=sess.run([optimizer,error,accuracy],feed_dict={x:image,y_true:label})
            print("%d train,loss is %f accuracy is %f" % (i+1,loss,accuracy_value ))

    return None


if __name__=="__main__":
    full_connection()