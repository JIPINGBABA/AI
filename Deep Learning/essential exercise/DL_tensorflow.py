# import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.compat.v1.disable_eager_execution()
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()



def tensorflow_demo():
    """
    TensorFlow structure
    :return:
    """
    #original python
    a=2
    b=3
    c=a+b
    print("ordinary addition:\n",c)

    #TensorFlow implement addition
    a_t=tf.constant(2)
    b_t=tf.constant(3)
    c_t=a_t+b_t
    print("the result of the addition operation with TensorFlow:\n",c_t)

    #Open session
    with tf.Session() as sess:
        c_t_value=sess.run(c_t)
        print("c_t_value:\n",c_t_value)

    return None

def graph_demo():
    """
    TensorFlow structure
    :return:
    """
    #original python
    a=2
    b=3
    c=a+b
    print("ordinary addition:\n",c)

    #view default graph
    default_g=tf.get_default_graph()
    print("default_g:\n",default_g)
    #TensorFlow implement addition
    a_t=tf.constant(2)
    print("a_t:\n",a_t)
    # print("graph attribute of a_t:\n",a_t.graph)
    b_t=tf.constant(3)
    print("b_t:\n",b_t)
    # print("graph attribute of b_t:\n", a_t.graph)
    c_t=a_t+b_t
    print("graph attribute of c_t:\n", a_t.graph)
    print("the result of the addition operation with TensorFlow:\n",c_t)

    #Open session
    with tf.Session() as sess:
        c_t_value=sess.run(c_t)
        print("graph attribute of session:\n", a_t.graph)
        print("c_t_value:\n",c_t_value)
        #write the graph locally to generate the event file
        tf.summary.FileWriter("./tmp/summary",graph=sess.graph)

    # user-defined graph
    new_g = tf.Graph()
    # define data and operations in my own graph
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new:\n", c_new)
    #open session
    with tf.Session(graph=new_g) as new_sess:
        c_new_value =new_sess.run(c_new)
        print("graph attribute of session:\n", a_new.graph)
        print("c_new_value:\n", c_new_value)
    return None
def session_demo():
    """
    session demo
    :return:
    """

    #TensorFlow implement addition
    a_t=tf.constant(2)
    print("a_t:\n",a_t)
    b_t=tf.constant(3)
    print("b_t:\n",b_t)
    c_t=a_t+b_t
    print("the result of the addition operation with TensorFlow:\n",c_t)
    #define placeholder
    a_ph=tf.placeholder(tf.float32)
    b_ph=tf.placeholder(tf.float32)
    c_ph=tf.add(a_ph,b_ph)
    print("a_ph:\n",a_ph)
    print("b_ph:\n",b_ph)
    print("c_ph:\n",c_ph)

    # view default graph
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)
    #Open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
        # run placeholder
        c_ph_value=sess.run(c_ph,feed_dict={a_ph:3.9,b_ph:4.8})
        print("c_ph_value:\n",c_ph_value)
        # a,b,c=sess.run((a_t,b_t,c_t))
        # print("abc:\n",a,b,c)
        # c_t_value=sess.run(c_t)
        # print("graph attribute of session:\n", a_t.graph)
        # print("c_t_value:\n",c_t_value)
        #write the graph locally to generate the event file
        tf.summary.FileWriter("./tmp/summary",graph=sess.graph)
    return None

def tensor_demo():
    """
    tensor demonstration
    :return:
    """
    tensor1=tf.constant(4.0)
    tensor2=tf.constant([1,2,3,4])
    linear_squares=tf.constant([[4],[9],[16],[25]],dtype=tf.int32)

    print("tensor1:\n",tensor1)
    print("tensor2:\n",tensor2)
    print("linear_squares_before:\n",linear_squares)

    #tensor type  change
    l_cast=tf.cast(linear_squares,dtype=tf.float32)
    print("linear_square_after:\n",linear_squares)
    print("l_cast:\n",l_cast)
    print("==================================================")
    #update change
    #define placeholder
    a_p=tf.placeholder(dtype=tf.float32,shape=[None,None])
    b_p=tf.placeholder(dtype=tf.float32,shape=[None,10])
    c_p=tf.placeholder(dtype=tf.float32,shape=[3,2])
    print("a_p:\n",a_p)
    print("b_p:\n",b_p)
    print("c_p:\n",c_p)
    #update the part of shape is undetermined
    a_p.set_shape([2,3])
    b_p.set_shape([2,10])

    #dynamic shape modification
    a_p_reshape=tf.reshape(a_p,shape=[2,3,1])
    print("a_p:\n", a_p)
    print("a_p_reshape:\n",a_p_reshape)


    print("a_p:\n", a_p)
    print("b_p:\n", b_p)
    return None

def variable_demo():
    """
    variable demonstration
    :return:
    """
    #create variable
    with tf.variable_scope("my_scope"):
        a=tf.Variable(initial_value=50)
        b=tf.Variable(initial_value=40)
    with tf.variable_scope("your_scope"):
        c=tf.add(a,b)
    print("a:\n",a)
    print("b:\n",b)
    print("c:\n",c)

    #initializer variable
    init=tf.global_variables_initializer()

    #open session
    with tf.Session() as sess:
        #initializer
        sess.run(init)
        a_value,b_value,c_value=sess.run([a,b,c])
        print("a_value:\n",a_value)
        print("b_value:\n",b_value)
        print("c_value:\n",c_value)

    return None

def linear_regression():
    """
    implement linear regression
    :return:
    """
    with tf.variable_scope("prepare_data"):
        #1.prepare dataset
        X=tf.random_normal(shape=[100,1],name="feature")
        y_true=tf.matmul(X,[[0.8]])+0.7
    with tf.variable_scope("create_model"):
        #2.build model
        #define model parameters
        weights=tf.Variable(initial_value=tf.random_normal(shape=[1,1]),name="Weights")
        bias=tf.Variable(initial_value=tf.random_normal(shape=[1,1]),name="Bias")
        y_predict=tf.matmul(X,weights)+bias
    with tf.variable_scope("loss_function"):
        #3.build loss function
        error=tf.reduce_mean(tf.square(y_predict-y_true))

    with tf.variable_scope("optimizer"):
        #4.optimize loss
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    #2_collect variables
    tf.summary.scalar("error",error)
    tf.summary.histogram("weights",weights)
    tf.summary.histogram("bias",bias)

    #3 merge variables
    merged=tf.summary.merge_all()
    #create Saver object
    saver=tf.train.Saver()


    #global variables initializer
    init=tf.global_variables_initializer()
    #open session
    with tf.Session() as sess:
        #initial variables
        sess.run(init)
        #1_create event file
        file_writer=tf.summary.FileWriter("./tmp/linear",graph=sess.graph)


        #check the values after initializing the model
        print("before train the parameters of model: weights%f,bias%f,loss%f" % (weights.eval(),bias.eval(),error.eval()))
        #begin train
        # for i in range(100):
        #     sess.run(optimizer)
        #     print("after%d train the parameters of model: weights%f,bias%f,loss%f" % (i+1,weights.eval(), bias.eval(), error.eval()))
        #
        #     #run merged variable operation
        #     summary=sess.run(merged)
        #     #4_write the variables after each iteration to the event
        #     file_writer.add_summary(summary,i)
        #
        #     #save model
        #     if i%10:
        #         saver.save(sess,"./tmp/model/my_linear.ckpt")
        # load model
        if os.path.exists("./tmp/model/checkpoint"):
            saver.restore(sess,"./tmp/model/my_linear.ckpt")
        print("after train the parameters of model: weights%f,bias%f,loss%f" % ( weights.eval(), bias.eval(), error.eval()))
    return None

#1.define command
tf.app.flags.DEFINE_integer("max_step",100,"train model steps")
tf.app.flags.DEFINE_string("model_dir","Unknown","model directory and name")

#2.simplify variable name
FLAGS=tf.app.flags.FLAGS
def command_demo():
    """
    command parameter
    :return:
    """
    print("max_step:\n",FLAGS.max_step)
    print("model_dir:\n",FLAGS.model_dir)

    return None
def main(argv):
    print(argv)
    return None

if __name__ == '__main__':
    #1.TensorFlow structure
    # tensorflow_demo()
    #2.graph
    # graph_demo()
    #3.session
    # session_demo()
    # 4.demonstration
    # tensor_demo()
    #5.variable demonstration
    # variable_demo()
    #6.implement linear regression
    # linear_regression()
    #7.command parameter
    # command_demo()
    tf.app.run()

