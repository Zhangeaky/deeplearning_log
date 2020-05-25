import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
#创建直线 Y=2X+4 的数据集 10001个点
trX = np.linspace(-1, 1, 10001)
trY = 2*trX + np.ones(*trX.shape)*4 +\
    np.random.randn(*trX.shape)*0.03

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(0.0, name="weights")#权重
b = tf.Variable(0.0, name="bias")#偏移
# 线性回归的模型
# y_model = X*w +b(将数据集中的X带入选择好的w和b,算出y_model)
y_model = tf.multiply(X, w) + b

# 代价函数
cost = tf.square(Y - y_model)

# 梯度下降优化器,最小化代价函数
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

w_ = sess.run(w)
b_ = sess.run(b)

print("result Y="+str(w_)+"trX +"+str(b_))

# plt.figure(1)
# plt.plot(trX, trY, 'o')
#
# plt.xlabel('trX')
# plt.ylabel('trY')
#
# plt.show()


