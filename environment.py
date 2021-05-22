import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GridMap:
	# If is_random_map is true, requires kw argument map_size_x and map_size_y
	def __init__(self, init_loc, is_random_map, **kwargs):
		assert (len(init_loc) == 2), '2D initial location is not valid'
		self.cur_loc = np.array([init_loc[0], init_loc[1]])
		if is_random_map:
			assert ('map_size_x' in kwargs and 'map_size_y' in kwargs), 'Please define map size for random map'
			self.map_size = (kwargs['map_size_x'], kwargs['map_size_y'])
			self.map = np.random.randint(2, size=self.map_size)
		else:
			self.map = plt.imread('img1.png')
			self.map = (self.map[:,:,0]*0.299) + (self.map[:,:,1]*0.587) + (self.map[:,:,2]*0.144)
			self.map = (self.map>0.6)
			self.map_size = (self.map.shape[0], self.map.shape[1])
		self.padded_map = np.pad(self.map, 1, mode='wrap')
		self.map = np.concatenate((self.map[:,:,None],self.map[:,:,None],self.map[:,:,None]), axis=2)
		
		self.cur_loc_tile = self.map[self.cur_loc[0], self.cur_loc[1], :].copy()
		self.cur_loc_color = np.array([1,0,0])
		self.map[self.cur_loc[0], self.cur_loc[1],:] = self.cur_loc_color
		
		self.actual_map_render = None
		self.slam_map_render = None
	
	
	def reset(self):
		return None
	
	def show_actual_map(self):
		fig, ax = plt.subplots()
		ax.set_title("Actual Map")
		self.actual_map_render = ax.imshow(self.map.astype('float'), cmap='gray')
		plt.ion()
		plt.show()
		
	def show_slam_map(self, slam_map):
		fig, ax = plt.subplots()
		ax.set_title("SLAM Map")
		self.slam_map_render = ax.imshow(slam_map, cmap='gray')
		fig.colorbar(self.slam_map_render)
		plt.ion()
		plt.show()
		

	def render(self, slam_map, pred_loc):
		self.map[self.cur_loc[0], self.cur_loc[1], :] = self.cur_loc_color
		self.actual_map_render.set_data(self.map.astype('float'))
		slam_map_loc = np.concatenate((slam_map[:,:,None],slam_map[:,:,None],slam_map[:,:,None]), axis=2)
		slam_map_loc[int(pred_loc[0]), int(pred_loc[1]), :] = np.array([1,0,0])
		self.slam_map_render.set_data(slam_map_loc)
		plt.draw()
		
	def step(self, action):
		observation = self.process_input_action(action)
		return observation
		
	def process_input_action(self, action):
		move_prob = [0.8, 0.1, 0.1]
		self.map[self.cur_loc[0], self.cur_loc[1], :] = self.cur_loc_tile
		possible_move = [np.array([0,0]), np.array([0,0]), np.array([0,0])]
		if action == 'w':
			possible_move = [np.array([-1,0]), np.array([-1,-1]), np.array([-1,1])]
		elif action == 'd':
			possible_move = [np.array([0,1]), np.array([-1,1]), np.array([1,1])]
		elif action == 's':
			possible_move = [np.array([1,0]), np.array([1,-1]), np.array([1,1])]
		elif action == 'a':
			possible_move = [np.array([0,-1]), np.array([-1,-1]), np.array([1,-1])]
		else:
			print("Please enter a valid move")
			return None
		self.cur_loc += possible_move[np.random.choice([0,1,2], p=move_prob)]
		self.cur_loc[0] = self.cur_loc[0] % self.map_size[0]
		self.cur_loc[1] = self.cur_loc[1] % self.map_size[1]
		self.cur_loc_tile = self.map[self.cur_loc[0], self.cur_loc[1], :].copy()
		
		observation = self.get_observation()
		return observation
		

	def get_observation(self):
		obs = self.padded_map[self.cur_loc[0]:self.cur_loc[0]+3, self.cur_loc[1]:self.cur_loc[1]+3].copy()
		return obs
		
	def get_map_size(self):
		return self.map_size
