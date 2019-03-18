import numpy as np
import tensorflow as tf
from models import CNNModel
from config import config
import os
import pandas as pd
#import matplotlib.pyplot as plt
from data import DataLoader
from scipy.stats.stats import pearsonr   


config=config()
data_loader = DataLoader(config)

print("Importing done")
sess = tf.Session()
model = CNNModel()
saver = tf.train.Saver()
print("Model made")

saver.restore(sess, config.logdir + '/' + config.restore_file)
print("Loading", config.logdir + '/' + config.restore_file, "Done")

xs, ys = data_loader.load_batch(config.batch_size, train=True)


num_sampels = len(xs)
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))))
val_loss = loss.eval(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
print("Val Loss", val_loss)
y_hats = model.y.eval(session=sess, feed_dict={model.x: xs,  model.keep_prob: 1.0})
print(ys.shape, y_hats.shape)
np.savetxt('pred.txt', np.array([np.squeeze(ys), np.squeeze(y_hats)]))
print(ys[0], y_hats[0])
