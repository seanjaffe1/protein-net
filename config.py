class config(object):
	def __init__(self):
		self.data_dir = 'data/test_data'
		self.data_file = 'scores.csv'
		self.restore_dir = None
		self.restore_file = None
		self.logdir = "./train_model_test"
        
		self.batch_size = 1 
		self.train_percent = 0.8
		self.learning_rate = .001
		self.l2_reg = 0.0
		self.keep_prob = 1.0
		self.num_steps = 1000
		self.keep_prob = 1
