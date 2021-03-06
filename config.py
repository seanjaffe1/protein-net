class config(object):
	def __init__(self):
		self.data_dir = 'data/human'
		self.data_file = 'human_scores.csv'
		self.restore_file = 'model-step-460-val-400659.ckpt'
		self.logdir = "./logs/model9"
		self.checkpoint_every = 50
		self.log = True
		self.modelno = 9
        
		self.batch_size = 100
		self.train_percent = 0.8
		self.learning_rate = .002
		self.l2_reg = 0.0
		self.keep_prob = 1.0
		self.num_steps = 600
		self.keep_prob = 1
