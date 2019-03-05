import random
import pandas as pd
import numpy as np

class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.load()
        
        
    def load(self):
        xs = []
        ys = []
        
        self.train_batch_pointer = 0
        self.test_batch_pointer = 0
        
        total = 0
        
        # Load data from wherever TODO
        with open(self.config.data_dir + '/' + self.config.data_file) as f:
            db = pd.read_csv(curr_data_file, index_col=0)
            
        for i in range(1000):
            c = np.random.random_sample((100, 24))
            xs.append(c)
            ys.append(.9)
        print("Generated Data!")
        
        self.num_sampels = len(xs)
        
        d = list(zip(xs, ys))
        random.shuffle(d)
        xs, ys = zip(*d)
        
        self.train_xs = xs[:int(len(xs) * self.config.train_percent)]
        self.train_ys = ys[:int(len(ys) * self.config.train_percent)]
        
        self.test_xs = xs[int(len(xs) * (1-self.config.train_percent)):]
        self.test_ys = ys[int(len(ys) * (1-self.config.train_percent)):]
        
        self.num_train_samples = len(self.train_xs)
        self.num_test_samples = len(self.test_xs)
        
    # Load a single batch
    def load_batch(self, batch_size, train=True):
        x_out = []
        y_out = []
        if train:
            for i in range(0, batch_size):
                # TODO load pdb data here
                
                x_out.append(self.train_xs[(self.train_batch_pointer + i) % self.num_train_samples])
                y_out.append(self.train_ys[(self.train_batch_pointer + i) % self.num_train_samples])
            self.train_batch_pointer += batch_size
        else:
            for i in range(0, batch_size):
            # TODO load pdb data here
                x_out.append(self.test_xs[self.test_batch_pointer + i % self.num_test_samples])
                y_out.append(self.test_ys[self.test_batch_pointer + i % self.num_test_samples])
            self.test_batch_pointer += batch_size

        x_out = np.array(x_out).reshape(self.config.batch_size, 100, 24, 1)
        y_out = np.array(y_out).reshape(self.config.batch_size, 1)
        return x_out, y_out
    

    def skip(self, num):
        self.train_batch_pointer += num
            
        
                          