import os
import tensorflow as tf
from data import DataLoader
from models import CNNModel
from config import config
import argparse

config = config()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", help="Continue training from a previous run", action='store_true')
    parser.add_argument("--restore_dir", help="Directory to restore values from", default=config.restore_dir)
    parser.add_argument("--storemetadata", help="Store metadata for tensorboard", action='store_true')
    parser.add_argument("--logdir", type=str, default=config.logdir, help="Directory for log files")
    return parser.parse_args()


def main():
    args = get_arguments()
                        
    sess = tf.Session()
    
    model = CNNModel()
    train_vars = tf.trainable_variables()
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * config.l2_reg
    train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
    
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    start_step = 0
    if args.restore:
        saver.restore(sess, args.logdir + '/' + args.restore_from)
        start_step = float(args.restore_from.split('step-')[0].split('-')[-1])
        print('Restored Model from '+ args.log_dir + '/'+ args.restore_from)
        
    if args.storemetadata:
        tf.scalar_summary("loss", loss)
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(args.logdir, graph=tf.get_default_graph())
        
    min_loss = 1.0
    data_loader = DataLoader(config)
    
    for i in range(start_step, start_step + config.num_steps):
        xs, ys = data_loader.load_batch(config.batch_size, train=True)
        train_step.run(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: config.keep_prob})

        train_error = loss.eval(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        print("Step %d, train loss %g" % (i, train_error))

        '''
        TODO: add extra logging
        if i % 10 == 0:
            xs, ys = data_loader.load_batch(args.batch_size, train=True)
            val_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: config.1.0})
            print("Step %d, train loss %g" % (i, train_error))
        '''
if __name__ == '__main__':
    main()