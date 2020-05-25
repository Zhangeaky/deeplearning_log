import tensorflow as tf
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[4, 5], [6, 7]])
massage = tf.constant('Result of matrix')
product = tf.matmul(matrix1, matrix2)
sess = tf.Session()
result = sess.run(product)
# print(sess.run(massage))
# print(result)
#sess.close()

op1 = tf.constant([[1, 1], [2, 2], [0, 0]])
op2 = tf.constant([[1, 2], [3, 4], [5, 6]])
res = tf.add(op1, op2)
op1 = sess.run(res)
print(op1)

