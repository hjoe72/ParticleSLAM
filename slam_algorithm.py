import numpy as np

class ParticleSLAM:
	def __init__(self, map_size_x, map_size_y, n_particles=100):
		self.n_particles = n_particles
		self.map_size_x = map_size_x
		self.map_size_y = map_size_y
		self.pred_map = np.zeros((map_size_x+2, map_size_y+2))
		self.particles_loc = np.zeros((n_particles,2))
		self.particles_loc[:,0] += int(map_size_x/2)
		self.particles_loc[:,1] += int(map_size_y/2)
		self.alpha = np.ones(n_particles)*(1/n_particles)
		self.N_thres = self.n_particles*0.3
		
		print("Created Particle SLAM with {} particles.".format(n_particles))
		
	def predict(self, action):
		move_prob = [0.8, 0.1, 0.1]
		possible_move = [np.array([0,0]), np.array([0,0]), np.array([0,0])]
		if action == 'w':
			possible_move = [np.array([-1,0]), np.array([-1,-1]), np.array([-1,1])]
		elif action == 'd':
			possible_move = [np.array([0,1]), np.array([-1,1]), np.array([1,1])]
		elif action == 's':
			possible_move = [np.array([1,0]), np.array([1,-1]), np.array([1,1])]
		elif action == 'a':
			possible_move = [np.array([0,-1]), np.array([-1,-1]), np.array([1,-1])]
			
		for i in range(self.n_particles):
			self.particles_loc[i,:] += possible_move[np.random.choice([0,1,2], p=move_prob)]
			if self.particles_loc[i,0] < 1:
				self.particles_loc[i,0] = self.map_size_x
			elif self.particles_loc[i,0] > self.map_size_x:
				self.particles_loc[i,0] = 1
			if self.particles_loc[i,1] < 1:
				self.particles_loc[i,1] = self.map_size_y
			elif self.particles_loc[i,1] > self.map_size_y:
				self.particles_loc[i,1] = 1
			
			
	def update(self, observation):
		observation = ((observation.astype('float')*2)-1)
		corr = np.zeros(self.n_particles)
		for i in range(self.n_particles):
			loc_x = int(self.particles_loc[i,0])
			loc_y = int(self.particles_loc[i,1])
			corr[i] = np.sum(np.multiply(observation, self.pred_map[loc_x-1:loc_x+2, loc_y-1:loc_y+2]))
		
		max_corr_ind = np.argmax(corr)
		loc_x_pred = int(self.particles_loc[max_corr_ind,0])
		loc_y_pred = int(self.particles_loc[max_corr_ind,1])
		self.pred_map[loc_x_pred-1:loc_x_pred+2, loc_y_pred-1:loc_y_pred+2] += observation*1.4
			
		self.pred_map = np.clip(self.pred_map, -8, 8)
		
		softmax_val = self.softmax(corr)
		for i in range(self.n_particles):
			self.alpha[i] = softmax_val[i]
			
		N_eff = 1/np.sum(np.square(self.alpha))
		if N_eff < self.N_thres:
			self.resample_particles()
		
		
	def resample_particles(self):
		resample_ind = np.random.choice(range(self.n_particles), size=self.n_particles, p=self.alpha)
		new_particles_loc = np.zeros((self.n_particles, 2))
		
		for i in range(self.n_particles):
			new_particles_loc[i,:] = self.particles_loc[resample_ind[i],:].copy()
			
		self.particles_loc = new_particles_loc.copy()
		self.alpha = np.ones(self.n_particles)*(1/self.n_particles)
		print("Particles resampled")
			
		
	def softmax(self, x):
		return np.exp(x)/np.sum(np.exp(x))
			
	def sigmoid(self, x):
		return 1/(np.exp(-1*(x.astype('float')))+1)
	
	def get_map_image(self):
		return self.sigmoid(self.pred_map)
		
	def get_pred_loc(self):
		max_corr_ind = np.argmax(self.alpha)
		return (self.particles_loc[max_corr_ind,0], self.particles_loc[max_corr_ind,1])
