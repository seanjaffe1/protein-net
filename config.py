class config(object):
	def __init__(self):
		self.data_dir = 'data/cameo_data'
		self.data_file = 'scores.csv'
		self.restore_file = 'model-step-990-val-107182.ckpt'
		self.logdir = "./train_model_test"
		self.checkpoint_every = 5
		self.log = True
        
		self.batch_size = 100
		self.train_percent = 0.8
		self.learning_rate = .002
		self.l2_reg = 0.0
		self.keep_prob = 1.0
		self.num_steps = 1200
		self.keep_prob = 1
