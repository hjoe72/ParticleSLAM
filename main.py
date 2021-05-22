import environment
import slam_algorithm
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--random_map', dest='is_random_map', action='store_true')
parser.add_argument('--num_particles', dest='num_particles', type=int, default=100)

args = parser.parse_args()

init_loc = (12,10)

if args.is_random_map:
	map_size_x = 24
	map_size_y = 20
	env = environment.GridMap(init_loc, args.is_random_map, map_size_x=map_size_x, map_size_y=map_size_y)
else:
	env = environment.GridMap(init_loc, args.is_random_map)
slam = slam_algorithm.ParticleSLAM(env.get_map_size()[0], env.get_map_size()[1], args.num_particles)

observation = env.reset()


env.show_slam_map(slam.get_map_image())
env.show_actual_map()

for i in range(500):
	env.render(slam.get_map_image(), slam.get_pred_loc())
	action = input("Enter Action 'W', 'A', 'S', or 'D' followed by Enter Key \n")
	observation = env.step(action)
	if observation is not None:
		slam.predict(action)
		slam.update(observation)
		
	
	
	
