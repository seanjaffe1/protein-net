import tensorflow as tf

def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initial = initializer(shape=shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class CNNModel9(object):
    
    def __init__(self, dropout_prob=0.0, batch_norm=False, is_trianing=True):
        x = tf.placeholder(tf.float32, shape=[None, None, 24, 1], name='x')
        y_= tf.placeholder(tf.float32, shape=[None, 1])
        
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        x_val = x
        
        self.W_conv1 = weight_variable([5, 24, 1, 16])  #h, w, #in channels, #out channels
        self.b_conv1 = bias_variable([16])
        self.h_conv1 = tf.nn.relu(conv2d(x_val, self.W_conv1,3) + self.b_conv1)
        
        self.W_conv2 = weight_variable([5, 1, 16, 20])
        self.b_conv2 = bias_variable([20])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2,3) + self.b_conv2)
                
        #TODO why 1840?
        self.W_fc1 = weight_variable([886000, 1164]) # in, out
        self.b_fc1 = bias_variable([1164])
        
        self.h_conv2_flat = tf.reshape(self.h_conv2, [-1,886000]) 
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv2_flat, self.W_fc1) + self.b_fc1)
        
        self.W_fc2 = weight_variable([1164, 100])
        self.b_fc2 = bias_variable([100])
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2, name='fc2')
        
        #if batch_norm:
        #    self.h_fc2 = tf.contrib.layers.batch_norm(self.h_fc2, is_training=is_training, trainable=True)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob)

        self.W_fc3 = weight_variable([100, 50])
        self.b_fc3 = bias_variable([50])
        self.h_fc3 = tf.nn.relu(tf.matmul(self.h_fc2_drop, self.W_fc3) + self.b_fc3, name='fc3')
        #if batch_norm:
        #    self.h_fc3 = tf.contrib.layers.batch_norm(self.h_fc3, is_training=is_training, trainable=True)
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, keep_prob)

        self.W_fc4 = weight_variable([50, 10])
        self.b_fc4 = bias_variable([10])
        self.h_fc4 = tf.nn.relu(tf.matmul(self.h_fc3_drop, self.W_fc4) + self.b_fc4, name='fc4')
        #if batch_norm:
        #    self.h_fc4 = tf.contrib.layers.batch_norm(self.h_fc4, is_training=is_training, trainable=True)
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4, keep_prob)

        self.W_fc5 = weight_variable([10, 1])
        self.b_fc5 = bias_variable([1])
        y = tf.add(tf.matmul(self.h_fc4_drop, self.W_fc5), self.b_fc5, name='y') #todo change to relu

        self.x = x
        self.y_ = y_
        self.y = y
        self.keep_prob = keep_prob
        self.fc2 = self.h_fc2
        self.fc3 = self.h_fc3
