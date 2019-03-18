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
    #parser.add_argument("--restore_dir", help="Directory to restore values from", default=config.restore_dir)
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
        saver.restore(sess, args.logdir + '/' + config.restore_file)
        start_step = int(config.restore_file.split('step-')[1].split('-')[0])
        print(os.getcwd())
        print('Restored Model from '+ config.logdir + '/'+ config.restore_file)
        
    if args.storemetadata:
        tf.scalar_summary("loss", loss)
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(args.logdir, graph=tf.get_default_graph())
     
    min_train_loss = 999999999.0
    min_val_loss = 999999999.0
    data_loader = DataLoader(config)
    
    for i in range(start_step, start_step + config.num_steps):
        xs, ys = data_loader.load_batch(config.batch_size, train=True)
        train_step.run(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: config.keep_prob})
        
        train_error = loss.eval(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        min_train_loss = min(min_train_loss, train_error)
        print("Step %d, train loss %g" % (i, train_error))
        preds =model.y.eval(session=sess, feed_dict={model.x: xs,  model.keep_prob: 1.0})
        #for i in range(10):
        #    print(preds[i], ys[i])
        
        
        if i%10==0:
            xs_t, ys_t = data_loader.load_batch(config.batch_size, train=False)
            val_loss = loss.eval(session=sess, feed_dict={model.x: xs_t, model.y_: ys_t, model.keep_prob: 1.0})
            print("Val Loss", val_loss)
            
            if i > 0 and i % config.checkpoint_every == 0 and config.log:
                print("Logging")
                if not os.path.exists(args.logdir):
                    #print("Logging 1")
                    os.makedirs(args.logdir)
                    checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_loss))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)
                elif val_loss < min_val_loss:
                    #print("Logging 2")
                    min_val_loss = val_loss
                    if not os.path.exists(args.logdir):
                        os.makedirs(args.logdir)
                    checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_loss))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)
                    
                    
                    
    xs, ys = data_loader.load_batch(config.batch_size, train=True)
    xs_t, ys_t = data_loader.load_batch(config.batch_size, train=False)
    val_loss = loss.eval(session=sess, feed_dict={model.x: xs_t, model.y_: ys_t, model.keep_prob: 1.0})
    min_val_loss = min(val_loss, min_val_loss)
    print("Min Train Loss:", min_train_loss, "Min Val Loss", min_val_loss)
    
if __name__ == '__main__':
    main()