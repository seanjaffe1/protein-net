import os
import tensorflow as tf
from data import DataLoader
from models import *
from config import config
import argparse
import json
import numpy as np
import time


def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
config = config()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", help="Continue training from a previous run", action='store_true')
    #parser.add_argument("--restore_dir", help="Directory to restore values from", default=config.restore_dir)
    parser.add_argument("--storemetadata", help="Store metadata for tensorboard", action='store_true')
    parser.add_argument("--logdir",  default=config.logdir, help="Directory for log files")
    parser.add_argument("--modelno", default=config.modelno, type=int, help="Model number to use")
    return parser.parse_args()


def main():
    args = get_arguments()
                        
    sess = tf.Session()
    
    model = None
    if args.modelno == 0:
            model = CNNModel()
    elif args.modelno == 1:
            model = CNNModel1()
    elif args.modelno == 2:
            model = CNNModel2()
    elif args.modelno == 3:
            model = CNNModel3()
    elif args.modelno == 4:
            model = CNNModel4()
    elif args.modelno == 10:
            model = CNNModel10()
    # size 4000 models
    elif args.modelno == 9:
            model = CNNModel9()
    else:
        print("Invalide model no")
        raise
        
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
        print('Restored Model from '+ args.logdir + '/'+ config.restore_file)
        
    if args.storemetadata:
        tf.scalar_summary("loss", loss)
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(args.logdir, graph=tf.get_default_graph())
    
    
    train_losses = []
    val_losses = []
    errors = []
    
    min_train_loss = 999999999.0
    min_val_loss = 999999999.0
    data_loader = DataLoader(config)
    
    start_time = time.time()
    for i in range(start_step, start_step + config.num_steps):
        xs, ys = data_loader.load_batch(config.batch_size, train=True)
        train_step.run(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: config.keep_prob})
        
        train_error = loss.eval(session=sess, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})  
        train_losses.append(train_error)
        min_train_loss = min(min_train_loss, train_error)
        print("Step %d, train loss %g" % (i, train_error))
        preds =model.y.eval(session=sess, feed_dict={model.x: xs,  model.keep_prob: 1.0})
        #for i in range(10):
        #    print(preds[i], ys[i])
        if config.modelno >= 10:
            mean_error = mean_absolute_percentage_error(np.array(ys[:,0], dtype=float), np.array(preds[:,0], dtype=float))
        else:
            mean_error = mean_absolute_percentage_error(np.array(ys, dtype=float), np.array(preds, dtype=float))


        errors.append(mean_error)
        print("Step %d, train loss %g, mean error %g" % (i, train_error, mean_error))


        if i%10==0:
            xs_t, ys_t = data_loader.load_batch(config.batch_size, train=False)
            val_loss = loss.eval(session=sess, feed_dict={model.x: xs_t, model.y_: ys_t, model.keep_prob: 1.0})
            print("Val Loss", val_loss)
            val_losses.append(val_loss)
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
                    
                    
    time_elapsed = time.time() - start_time 
    print("Time Elapased", time_elapsed)
    xs, ys = data_loader.load_batch(config.batch_size, train=True)
    xs_t, ys_t = data_loader.load_batch(100, train=False)
    val_loss = loss.eval(session=sess, feed_dict={model.x: xs_t, model.y_: ys_t, model.keep_prob: 1.0})
    preds = model.y.eval(session=sess, feed_dict={model.x: xs_t, model.keep_prob: 1.0})
    '''
    if config.modelno == 10:
        mean_error = mean_absolute_error(np.array(preds[:,0], dtype=float), np.array(ys[:,0], dtype=float))
        print(len(ys[:,0]))
    else:
        mean_error = mean_absolute_error(np.array(preds, dtype=float), np.array(ys, dtype=float))
    errors.append(mean_error)
    '''

    min_val_loss = min(val_loss, min_val_loss)
    print("Min Train Loss:", min_train_loss, "Min Val Loss", min_val_loss)
    
    log_obj = {'train_losses': str(train_losses), 
               'val_losses': str(val_losses),
               'val_labels': str(np.squeeze(ys_t).tolist()),
               'val_pred'  : str(np.squeeze(preds).tolist()),
               'errors' : str(np.squeeze(errors).tolist()),
               'time' : time_elapsed,
               'model_no': config.modelno
                 }
    with open(args.logdir + '/metadata.txt', 'w') as outfile:  
        json.dump(log_obj, outfile)
    
if __name__ == '__main__':
    main()