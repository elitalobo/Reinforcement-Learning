import matplotlib.pylab as plt
import random
optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(1,0),(1,0),(0,1),(0,1),(0,0)]]

def get_optimal_action(x,y):
	return optimal_policy[x][y]

def random_pick(some_list, probabilities):
	x = random.uniform(0, 1)
	cumulative_probability = 0.0
	for item, item_probability in zip(some_list, probabilities):
		cumulative_probability += item_probability
		if x < cumulative_probability: break
	return item,item_probability

def rotate_right(x,y):
	return (y,-x)

def rotate_left(x,y):
	return (-y,x)

actions = [(0,1),(-1,0),(1,0),(0,-1)]
action_probabilities = [0.25,0.25,0.25,0.25]
action_result = ["SC","VL","VR","S"]
#action_result_probabilities = [1.0,0.0,0.0,0.0] #test optimal policy
action_result_probabilities = [0.8,0.05,0.05,0.1]
import numpy as np

grid_rules = {}
grid_rules[(2,2)]="INF"
grid_rules[(3,2)]="INF"
grid_rules[(4,4)]= "GOAL"
grid_len = 5
rewards ={}
rewards[(4,2)]=-10
rewards[(4,4)]=+10
gamma=0.9
def is_outside_boundary_or_found_obstacle(x,y):
	if(grid_rules.get((x,y))=="INF" or (x>=grid_len or y>=grid_len or x<0 or y<0)):
		return True
	return False

def pick_action():
	action_picked = random_pick(actions,action_probabilities)
	return action_picked[0]

def pick_action_success():
	action_success = random_pick(action_result,action_result_probabilities)
	return action_success

def get_reward(x,y,t):
	if rewards.get((x,y))==None:
		return 0.0
	else:
		return float(np.power(gamma,t)*rewards[(x,y)]) 

def print_timestep(x,y):
	i=0
	j=0
	while i< grid_len:
		j=0
		while j < grid_len:
			if(i==x and j==y):
				print('R '),
			else:
				res = '0 '
				if rewards.get((i,j))!=None:
					res = str(rewards[(i,j)]) + " "
				print(res),
			j+=1
		print("\n"),
		i+=1
	print "\n"

def get_new_state(x,y,t,optimal=False,Action=None,Action_Success=None):
	orig_x = x
	orig_y = y
	if Action == None:
		action = pick_action()
	else:
		action = Action
	if optimal:
		action = get_optimal_action(x,y)
	if Action_Success==None:
		action_success = pick_action_success()
	else:
		action_success= Action_Success
	#print action 
	#print action_success
	#import ipdb; ipdb.set_trace()
	if action_success[0] == "VR":
		action = rotate_right(action[0],action[1])
		x+=action[0]
		y+=action[1]
	elif action_success[0] == "SC":
		x+=action[0]
		y+=action[1]
	elif action_success[0] == "VL":
		action = rotate_left(action[0],action[1])
		x+=action[0]
		y+=action[1]
	#if((x==3 and y==2) or (x==2 and y==2)):
	#	import ipdb; ipdb.set_trace() 
	if(x==4 and y==2 and get_reward(x,y,t)==0):
		import ipdb; ipdb.set_trace()		
	if(is_outside_boundary_or_found_obstacle(x,y)):
		return (orig_x,orig_y,get_reward(x,y,t),grid_rules.get((orig_x,orig_y))=="GOAL")
	else:
		return (x,y,get_reward(x,y,t),grid_rules.get((x,y))=="GOAL")

from scipy import stats	
def plot_histogram(title,y_title,x_title,arr,filename):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	n, bins, rectangles = ax.hist(arr, 50, color='yellow',edgecolor='black')
	#plt.text(-17,5000, r'$\mu=' + str(np.mean(arr)) + ',\ \sigma=' + str(np.std(arr)) + '$',bbox=dict(facecolor='red', alpha=0.5))
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.title(title)
	#plt.show()      
	plt.savefig(filename)
	plt.clf()

e=0.00000000000000001
def find_optimal_policy_through_value_iter():
	v = {}
	i=0
	j=0
	q = {}
	prev_actions = None
	current_actions = []
	while i < grid_len:
		j=0
		while j < grid_len:
			v[(i,j)]=[]
			v[(i,j)].append(rewards.get((i,j)) or  0.0)
			j=j+1
		i=i+1
	i=j=0
	flag=True
	cnt=1
	counter =0
	while flag==True:
		print "entered flag"
		similarity=0
		counter+=1
		current_actions = []
		i=0
		while i < grid_len:
			j=0
			while j < grid_len:
				if grid_rules.get((i,j))=="INF":
					j+=1
					continue
				if grid_rules.get((i,j))=="GOAL":
					v[(i,j)].append(gamma*v[(i,j)][cnt-1])
					j+=1
					continue
				k=0
				max_value =-1000000000
				optimal_action=0
				while k < len(actions):
					value=0.0
					l=0
					while l < len(action_result):
						x,y,new_reward,goal_flag = get_new_state(i,j,0,False,actions[k],(action_result[l],action_result_probabilities[l]))
						#if(goal_flag==False):
						value+= action_result_probabilities[l]*((rewards.get((i,j)) or  0.0)+ gamma*v[(x,y)][cnt-1])
						#else:
						#	value+= action_result_probabilities[l]*((rewards.get(i,j))) + gamma*v[(x,y)][0]
						l+=1
					if value > max_value:
						max_value = value
						optimal_action = k
					k+=1
				current_actions.append(optimal_action)
				v[(i,j)].append(max_value)
				print str(v[(i,j)][cnt]) +  " (" + str(i)+" , "+ str(j) + ") , " + str(optimal_action) +  " " + str(max_value) + " " + str(v[(i,j)][cnt-1])
				#import ipdb; ipdb.set_trace()
				if(abs(v[(i,j)][cnt-1]-max_value) < e):
					similarity+=1
				j+=1
			i+=1
		#import ipdb; ipdb.set_trace()
		cnt+=1
		print cnt
		print "similarity " + str(similarity)
		if similarity== (grid_len*grid_len)-len(grid_rules):
			flag=False
	return current_actions	


#result = find_optimal_policy_through_value_iter()
'''i=0
cnt=0
while i < grid_len:
	j=0
	while j < grid_len:
		if grid_rules.get((i,j))==None:
			print actions[result[cnt]],
			cnt+=1
		else:
			print grid_rules.get((i,j)),
		print " ",
		j+=1
	i+=1
	print "\n",
''' 

time_steps=0
trials =1
req_trials = 5000000
start_x=3
start_y=4
trial_rewards = []
apply_optimal_policy = False
cnt_21=0.0

if __name__=='__main__':
	
	while trials <= req_trials:
		REWARD = (rewards.get(start_x,start_y) or 0.0)
		flag=False
		x=start_x
		y=start_y
		time_steps=0
		print ("Time step: " + str( time_steps)+"\n"),
		#print_timestep(x,y),
		REWARD = 0.0	
		while(flag==False):
			
			#print("Time step: " + str(time_steps)+"\n")
			if(time_steps==11 and x==4 and y==2):
				cnt_21+=1
			x,y,new_reward,flag = get_new_state(x,y,time_steps,apply_optimal_policy)				
			#print_timestep(x,y)	
			REWARD += new_reward
			#print("reward for this step: " + str(new_reward) + " " + str(time_steps))
			time_steps+=1
		print("total_reward for trial " + str(trials) + " : " + str(REWARD)),
		trial_rewards.append(REWARD)
		trials+=1
	
	
	trial_rewards = np.array(trial_rewards)
	#print("Minimum discounted reward: " + str(np.min(trial_rewards)))
	#print("Maximum discounted reward: " + str(np.max(trial_rewards)))

        #print("Mean discounted reward discounted reward: " + str(np.mean(trial_rewards)))
        #print("Standard Deviation of discounted rewards: " + str(np.std(trial_rewards)))
	#trial_rewards = x / np.linalg.norm(trial_rewards)
	plot_histogram('Discounted Rewards','Frequency','Histogram of Discounted Rewards',trial_rewards,'DiscountedRewardsHistogram.png')
	#plot_histogram('Discounted Rewards on using Optimal Policy','Frequency','Histogram of Discounted Rewards on using Optimal Policy',trial_rewards,'DiscountedRewardsHistogramOptimal.png')
	print "\n"
	print("Probability that S_19=21 given that S_8=18: " + str(cnt_21/req_trials)) 
