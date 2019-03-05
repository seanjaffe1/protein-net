import random
import pandas as pd
import numpy as np


def amino_one_hot(seq):
    encodings = {'G':1, 'A':2, 'L':3, 'M':4, 'F':5, 'W':6, 'K':7, 'Q':8, 'E':9, 'S':10,
                 'P':11,'V':12,'I':13,'C':14,'Y':15,'H':16,'R':17,'N':18,'D':19,'T':0}
    encoding = np.zeros((len(seq), 22))
    for i in range(len(seq)):
        if seq[i] in encodings.keys():
            encoding[i][encodings[seq[i]]] = 1 
        else:
            encoding[i][21]=1
    return encoding

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
        
        # Load data
        db = pd.read_csv(self.config.data_dir + '/' + self.config.data_file, index_col=0)
        for i in db.index.values:
            curr_frame = db.iloc[i]
            total = curr_frame.total
            length = curr_frame.len
            angles = np.loadtxt(self.config.data_dir + '/' + curr_frame.pdb_id + '.angles.txt', delimiter=',')
            seq = amino_one_hot(curr_frame.seq)
            if np.isnan(angles).any() :
                continue
            
            xs.append([seq, angles, length])
            ys.append(total)

        
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
        
        print(str(self.num_train_samples), 'training samples')
        print(str(self.num_test_samples), 'testing samples')

    # Load a single batch
    def load_batch(self, batch_size, train=True):
        if train:
            return self.load_batch_s(batch_size, ptr=self.train_batch_pointer, xs=self.train_xs, ys=self.train_ys)
        else:
            return self.load_batch_s(batch_size, ptr=self.test_batch_pointer, xs=self.test_xs, ys=self.test_ys)
    
    def load_batch_s(self, batch_size, ptr, xs, ys, train=True):
        max_len = 0
        #a = [xs[i][2] for i in range(ptr % len(xs), (ptr  + batch_size)% len(xs))]
        #print(type(a), type(a[0]), a)
        '''
        if ptr % len(xs) < (ptr + batch_size) % len(xs):
            max_len = max(np.max([xs[i][2] for i in range(ptr % len(xs), len(xs))]), 
                                  np.max([xs[i][2] for i in range(0, (ptr % len(xs) + batch_size) % len(xs))]))
        else:
            print(ptr, len(xs), batch_size)
            max_len = np.max([xs[i][2] for i in range(ptr % len(xs), (ptr  + batch_size)% len(xs))])
        '''
        max_len=3612
        x_out = np.zeros((batch_size, int(max_len), 24))
        y_out = []

        for i in range(0, batch_size):
            # TODO load pdb data here
            x = xs[(ptr + i) % len(xs)]
            
            #Use the following when you figured out how to do variable sizes
            
            x_out[i,0:int(x[2]),:22] = x[0]
            x_out[i,0:int(x[2]), 22] = x[1][0]
            x_out[i,0:int(x[2]), 23] = x[1][1]
            
            
            y_out.append(ys[(ptr + i) % len(xs)])
        if train:
            self.train_batch_pointer += batch_size
        else:
            self.test_batch_pointer += batch_size

        x_out = np.array(x_out).reshape(self.config.batch_size, int(max_len), 24, 1)
        y_out = np.array(y_out).reshape(self.config.batch_size, 1)
        return x_out, y_out
    

    def skip(self, num):
        self.train_batch_pointer += num
            
        
                          