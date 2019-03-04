class config(object):
	def __init__(self):
		self.datda_dir = 'data'
		self.data_file = 'cameo_scores.csv'
		self.restore_dir = None
		self.restore_file = None
		self.logdir = "./train_model"
        
		self.batch_size = 10   
		self.train_percent = 0.8
		self.learning_rate = .001
		self.l2_reg = 0.0
		self.keep_prob = 1.0
		self.num_steps = 1000
		self.keep_prob = 1
