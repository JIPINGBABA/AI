import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


class Cifar():

    # def __init__(self):
    #
    #
    #     #initialize
    #     self.height = 32
    #     self.width = 32
    #     self.channels = 3
    #     #byte
    #     self.image_bytes=self.height * self.width * self.channels
    #     self.label_bytes=1
    #     self.all_bytes=self.label_bytes + self.image_bytes
    def __init__(self):

        # 设置图像大小
        self.height = 32
        self.width = 32
        self.channel = 3

        # 设置图像字节数
        self.image = self.height * self.width * self.channel
        self.label = 1
        self.sample = self.image + self.label


    def read_binary(self):
        """
        读取二进制文件
        :return:
        """
        # 1、构造文件名队列
        filename_list = os.listdir("./cifar-10-batches-bin")
        # print("filename_list:\n", filename_list)
        file_list = [os.path.join("./cifar-10-batches-bin/", i) for i in filename_list if i[-3:] == "bin"]
        # print("file_list:\n", file_list)
        file_queue = tf.train.string_input_producer(file_list)

        # 2、读取与解码
        # 读取
        reader = tf.FixedLengthRecordReader(self.sample)
        # key文件名 value样本
        key, value = reader.read(file_queue)

        # 解码
        image_decoded = tf.decode_raw(value, tf.uint8)
        print("image_decoded:\n", image_decoded)

        # 切片操作
        label = tf.slice(image_decoded, [0], [self.label])
        image = tf.slice(image_decoded, [self.label], [self.image])
        print("label:\n", label)
        print("image:\n", image)

        # 调整图像的形状
        image_reshaped = tf.reshape(image, [self.channel, self.height, self.width])
        print("image_reshaped:\n", image_reshaped)

        # 三维数组的转置
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0])
        print("image_transposed:\n", image_transposed)

        # 3、构造批处理队列
        image_batch, label_batch = tf.train.batch([image_transposed, label], batch_size=100, num_threads=2,
                                                  capacity=100)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            label_value, image_value = sess.run([label_batch, image_batch])
            print("label_value:\n", label_value)
            print("image:\n", image_value)

            coord.request_stop()
            coord.join(threads)

        return image_value, label_value

    def read_and_decode1(self,file_list):
        #1.build file name queue
        file_queue=tf.train.string_input_producer(file_list)
        #2.read anf decodeZ`
        #read
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        #key is filename value is a sample
        key,value=reader.read(file_queue)
        print("key:\n",key)
        print("value:\n",value)
        # decode
        decoded = tf.decode_raw(value, tf.uint8)
        print("decode:\n", decoded)

        #slice the target value and feature value
        label=tf.slice(decoded,[0],[self.label_bytes])
        image=tf.slice(decoded,[self.label_bytes],[self.image_bytes])
        print("label:\n",label)
        print("image:\n",image)

        #adjusting the image shape
        image_reshape=tf.reshape(image,shape=[self.channels,self.height,self.width])
        print("image_reshaped:\n",image_reshape)

        #transpose height,wigth,channel
        image_transposed=tf.transpose(image_reshape,[1,2,0])
        print("image_transposed:\n",image_transposed)

        #adjust image type
        image_cast=tf.cast(image_transposed,tf.float32)

        #3.batch
        label_batch,image_batch=tf.train.batch([label,image_cast],batch_size=100,num_threads=1,capacity=100)
        print("label_batch:\n",label_batch)
        print("image_batch:\n",image_batch)

        #  #open session
        with tf.Session() as sess:
            #start tread
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            key_new,value_new,decoded_new,label_new,image_new,image_reshape_new,image_transposed_new,label_value,image_value=sess.run([key,value,decoded,label,image,image_reshape,image_transposed,label_batch,image_batch])
            print("key_new:\n",key_new)
            print("value_new:\n",value_new)
            print("decoded_new:\n",decoded_new)
            print("label_new:\n",label_new)
            print("image_new:\n",image_new)
            print("image_reshaped_new:\n",image_reshape_new)
            print("image_transposed_new:\n",image_transposed_new)
            print("label_value:\n",label_value)
            print("image_value:\n",image_value)

            #collection
            coord.request_stop()
            coord.join(threads)
        return image_value,label_value
    # def write_to_tfrecords(self,image_batch,label_batch):
    #     """
    #    write the feature values and the target valuesof the sample to TFrecords file
    #     :param image:
    #     :param label:
    #     :return:
    #     """
    #     with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
    #         for i in range(100):
    #             #the loop constructs the example object and serializes it to a file
    #             image=image_batch[i].tostring()
    #             label=label_batch[i][0]
    #             # print("tfrecords_image:\n",image)
    #             # print("tfrecords_label:\n",label)
    #             example = tf.train.Example(features=tf.train.Features(feature={
    #                 "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    #                 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #             }))
    #
    #             writer.write(example.SerializeToString())
    #     return None
    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将样本的特征值和目标值一起写入tfrecords文件
        :param image:
        :param label:
        :return:
        """
        with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                print("tfrecords_image:\n", image)
                print("tfrecords_label:\n", label)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))

                # example.SerializeToString()
                # 将序列化后的example写入文件
                # writer.write(example.SerializeToString())
                writer.write(example.SerializeToString())
        print("!!!!!!!!!!!!")
        return None

    def read_tfrecords(self):
        """
        read TFRecords file
        :return:
        """
        # 1、build fielname queue
        file_queue = tf.train.string_input_producer(["cifar10.tfrecords"])

        # 2、read and decode
        # read
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # parse example
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = feature["image"]
        label = feature["label"]
        print("read_tf_image:\n", image)
        print("read_tf_label:\n", label)

        # decode
        image_decoded = tf.decode_raw(image, tf.uint8)
        print("image_decoded:\n", image_decoded)
        # 图像形状调整
        image_reshaped = tf.reshape(image_decoded, [self.height, self.width, self.channel])
        print("image_reshaped:\n", image_reshaped)

        #3、构造批处理队列
        image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size=100, num_threads=2, capacity=100)
        print("image_batch:\n", image_batch)
        print("label_batch:\n", label_batch)

        # open session
        with tf.Session() as sess:

            # thread
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            image_value,label_value=sess.run([image_batch,label_batch])
            print("image value:\n",image_value)
            print("label_value:\n",label_value)

            # image_value, label_value = sess.run([image_batch, label_batch])
            # print("image_value:\n", image_value)
            # print("label_value:\n", label_value)

            # colletion
            coord.request_stop()
            coord.join(threads)

        return None
if __name__=="__main__":
    file_name=os.listdir("./cifar-10-batches-bin")
    print("file_name:\n",file_name)
    #build filename list
    file_list=[os.path.join("./cifar-10-batches-bin/",file) for file in file_name if file[-3:]=="bin"]
    print("file_list:\n",file_list)

    #instance of Cifar
    cifar = Cifar()
    # image_value,label_value=cifar.read_and_decode(file_list)
    # image_value, label_value = cifar.read_binary()
    # cifar.write_to_tfrecords(image_value,label_value)
    cifar.read_tfrecords()



