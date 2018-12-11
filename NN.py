import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from DataProcessor import MData
import os


def model(ds, batch_size, epochs=200, lr_rate=0.01, batch_num=1):
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 4])
    y_ = tf.placeholder(tf.float32, [None, 3])

    W1 = tf.Variable(tf.truncated_normal([4, 10], stddev=0.1))
    b1 = tf.Variable(tf.zeros([10]))
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.1))
    b2 = tf.Variable(tf.zeros([3]))
    y = tf.nn.softmax(tf.matmul(y1, W2) + b2)
    loss = []
    acc = []

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
    # mse = tf.reduce_mean(tf.square(y-y_))

    my_opt = tf.train.GradientDescentOptimizer(lr_rate)
    train_step = my_opt.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epochs):
        for j in range(batch_num):
            train_data, train_lable = ds.get_batch(batch_size)
            sess.run(train_step, feed_dict={x: train_data, y_: train_lable})
            temp_loss = sess.run(cross_entropy, feed_dict={x: ds.train_data, y_: ds.train_lable})
            temp_acc = sess.run(accuracy, feed_dict={x: ds.train_data, y_: ds.train_lable})
        loss.append(temp_loss)
        acc.append(temp_acc)
        if i % 5 == 0:
            print('Step: ', i)
            print('Loss: ', temp_loss)
            # print(sess.run(W2, feed_dict={x: ds.train_data}))
            print('Accuracy: ', temp_acc)

    # prediction
    pred = sess.run(accuracy, feed_dict={x: ds.test_data, y_: ds.test_lable})
    print(pred)
    return loss, acc


def save_data(attr_name,vars):
    base_dir = './temp/' + attr_name + '/'
    try:
        os.makedirs(base_dir)
    except FileExistsError:
        pass

    iris = MData()
    iris.getdata('./iris.data.txt')
    for var in vars:
        loss, acc = model(iris, 3, 1000, 0.01, 45)
        with open(base_dir + str(var) + '_loss.txt', 'w') as loss_f:
            for item in loss:
                loss_f.write("%s\n" % item)
        loss_f.close()
        with open(base_dir + str(var) + '_acc', 'w') as acc_f:
            for item in acc:
                acc_f.write("%s\n" % item)
        acc_f.close()


if __name__ == '__main__':
    iris = MData()
    iris.getdata('./iris1.data.txt', 0.2)
    loss, acc = model(iris, 3, 1000, 0.01, 45)


    # fig, ax = plt.subplots()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.plot(loss_1, label='weight = 0')
    # plt.plot(loss_2, label='weight = 1')
    # plt.legend(loc='upper right')
    # plt.title('Loss Function on Batch Size')
    # plt.grid(True)
    # plt.show()