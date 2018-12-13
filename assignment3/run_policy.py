import matplotlib.pylab as plt
import random
optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(1,0),(1,0),(0,1),(0,1),(0,0)]]
import numpy as np
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
max_timesteps = 5000
gamma=0.9
exp_res = [np.power(gamma,num) for num in range(0,max_timesteps+100)]

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
value_function = None
def init_value_function():
	global value_function
	value_function = [0.0 for i in range(0,25)]

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
                return float(rewards[(x,y)])

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
	if(is_outside_boundary_or_found_obstacle(x,y)):
                return (orig_x,orig_y,get_reward(orig_x,orig_y,t),grid_rules.get((orig_x,orig_y))=="GOAL")
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
req_trials = 100
start_x = 0
start_y=0
trial_rewards = []
import sys
alphas = [np.power(10.0,-1*x) for x in np.arange(1,10,1)]
apply_optimal_policy = False
td_errors=[]

def plot_error(error,xlabel,ylabel,title,filename):
	global alphas
	new_alphas = [np.log10(x) for x in alphas]
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(np.array(new_alphas),np.array(error))
	#plt.show()
	plt.savefig(filename)
	plt.clf()

def run_episode(alpha,update_value_function=False,req_trials=100):
	trials = 0
	start_x=0
	global value_function
	start_y=0
	td_errors = []
	trial_rewards=[]
	while trials <= req_trials:
                        REWARD = ( 0.0)
                        flag=False
                        x=start_x
                        y=start_y
                        time_steps=0
                        #print ("Time step: " + str( time_steps)+"\n"),
                        #print_timestep(x,y),
                        REWARD = 0.0
                        prev_x = x
                        prev_y = y
			td_error = 0.0
                        while(flag==False):

                                #print("Time step: " + str(time_steps)+"\n")
                                x,y,new_reward,flag = get_new_state(x,y,time_steps,apply_optimal_policy)
				target = new_reward
				if flag==False:
					target=new_reward + gamma*value_function[x*grid_len+y]
				error = target - value_function[prev_x*grid_len+prev_y]
				if(update_value_function):	
                                	value_function[prev_x*grid_len+prev_y]=value_function[prev_x*grid_len+prev_y] + alpha*(error)
				td_errors.append(error)
                                #print_timestep(x,y)
				prev_x = x
				prev_y = y
                                REWARD += new_reward
                                #print("reward for this step: " + str(new_reward) + " " + str(time_steps))
                                time_steps+=1
                        #print("total_reward for trial " + str(trials) + " : " + str(REWARD)),
                   	trial_rewards.append(REWARD)
                        trials+=1
			#print time_steps
				
        trial_rewards = np.array(trial_rewards)
        #print("Minimum discounted reward: " + str(np.min(trial_rewards)))
        #print("Maximum discounted reward: " + str(np.max(trial_rewards)))

        #print("Mean discounted reward discounted reward: " + str(np.mean(trial_rewards)))
        #print("Standard Deviation of discounted rewards: " + str(np.std(trial_rewards)))
                	#trial_rewards = x / np.linalg.norm(trial_rewards)
                	#plot_histogram('Discounted Rewards','Frequency','Histogram of Discounted Rewards',trial_rewards,'DiscountedRewardsHistogram.png')
                	#plot_histogram('Discounted Rewards on using Optimal Policy','Frequency','Histogram of Discounted Rewards on using Optimal Policy',trial_rewards,'DiscountedRewardsHistogramOptimal.png')
        bellman_error = np.mean(td_errors)	
	errors = [np.power(x,2) for x in td_errors]
	return np.mean(errors),np.power(bellman_error,2.0)

if __name__=='__main__':
	
	lent = len(alphas)
	alpha_cnt =0
	td_errors = []
	bellman_errors = []
	while alpha_cnt < lent:
		init_value_function()
		run_episode(alphas[alpha_cnt],True,100)
		error,bellman_error = run_episode(alphas[alpha_cnt],False,100)
		td_errors.append(error)
		bellman_errors.append(bellman_error)
		print "TD error for alpha: " + str(alphas[alpha_cnt]) + " : " + str(error)
		alpha_cnt+=1
	print td_errors
	print bellman_errors
	plot_error(bellman_errors,"alphas","bellman_errors","bellman error vs alpha","bellman_error.png")
	plot_error(td_errors,"alphas","td_errors","td error vs alpha","final_gridworld_td_error.png")
	
