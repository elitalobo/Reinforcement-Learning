from numpy.random import randint
import matplotlib.pylab as plt
import random
s=[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,18,19,20,21,22,23,24]
optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(-1,0),(0,-1),(0,1),(0,1),(0,0)]]
#optimal_policy=[[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(0,1),(0,1),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(0,1),(-1,0),(0,0),(0,1),(1,0)],[(1,0),(1,0),(0,1),(0,1),(0,0)]]
import numpy as np
def get_optimal_action(x,y):
	return optimal_policy[x][y]

#best 0.01 -4.2
def random_pick(some_list, probabilities):
	x = random.uniform(0, 1)
	cnt=0
	cumulative_probability = 0.0
	for item, item_probability in zip(some_list, probabilities):
		cumulative_probability += item_probability
		if x < cumulative_probability: break
		cnt+=1
	try:
		assert(some_list[cnt]==item)
	except:
		return cnt-1
	return cnt

def rotate_right(x,y):
	return (y,-x)

def rotate_left(x,y):
	return (-y,x)
max_timesteps = 5000
gamma=0.9
ld=0.3
ldv=0
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
actions_len=4
rewards[(4,2)]=-10
rewards[(4,4)]=+10
gamma=0.9
epsilon = 0.1
alphas = [np.power(10.0,-1*x) for x in np.arange(1.0,7,0.5)]
e_traces = np.array([0.0 for x in range(0,100)])
value_function = np.array([0.0 for x in range(0,25)])
e_value = np.array([0.0 for x in range(0,25)])
q_value_function = np.array([0.0 for x in range(0,100)])
s_a_probs = np.array([0.25 for x in range(0,100)])

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

def softmax(x,epsilon):
    sigma = epsilon
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))*sigma)
    return (e_x) / e_x.sum()

def e_greedy(x):
	global epsilon
	probabilities = []
	maxm = np.max(x)
	cnt=0
	max_arr = []
	for value in x:
		if value == maxm:
			max_arr.append(cnt)
		cnt+=1
	idx = max_arr[0]
	if len(max_arr)!=1:	
		idx = max_arr[randint(0, len(max_arr)-1)]	
	cnt=0
	for value in x:
		if(cnt!=idx):
			probabilities.append(epsilon/actions_len)
		else:
			probabilities.append(1.0-epsilon+(epsilon/actions_len))
		cnt+=1
	return probabilities	
		
	
def initialize_q_value_function():
	global q_value_function
	q_value_function = np.array([0.0 for x in range(0,100)])
	e_traces = np.array([0.0 for x in range(0,100)])
	value_function = np.array([0.0 for x in range(0,25)])
	e_value = np.array([0.0 for x in range(0,25)])

def initialize_softmax_probs():
	global s_a_probs
	s_a_probs = np.array([0.25 for x in range(0,100)])

def initialize_epsilon_greedy_probs():
	global s_a_probs
	global q_value_functiion
	for x in range(0,25):
		req = e_greedy(q_value_function[x:x+4])
		s_a_probs[x*actions_len:x*actions_len+4]=req

def update_probabilities(epsilon=0.1,use_softmax=True):
	global s_a_probs
	for s in range(0,25):
		req_q = q_value_function[s*actions_len:s*actions_len+4]
		probs = None
		if use_softmax:
			probs = softmax(req_q,epsilon)
		else:
			probs = e_greedy(req_q)
		s_a_probs[s*actions_len:s*actions_len+4]=probs		
	return probs

def select_action(s):
	#import ipdb; ipdb.set_trace()
	global s_a_probs
	global actions
	#import ipdb; ipdb.set_trace()
	action_probs = s_a_probs[s*actions_len:s*actions_len+4]
	action_idx = random_pick(actions,action_probs)
	success_idx = random_pick(action_result,action_result_probabilities)
	return action_idx,success_idx


	
def get_reward(x,y,t):
        if rewards.get((x,y))==None:
                return 0.0
        else:
                return float(rewards[(x,y)])

def get_actual_reward(x,y,t):
	if rewards.get((x,y))==None:
                return 0.0
        else:
                return exp_res[t]*float(rewards[(x,y)])

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

def get_new_state(x,y,action_idx,success_idx,t):
	orig_x = x
	
	orig_y = y
	#import ipdb; ipdb.set_trace()
	#print action
	action = actions[action_idx]
	action_success = action_result[success_idx] 
	#print action_success 
	#import ipdb; ipdb.set_trace()
	if action_success == "VR":
		action = rotate_right(action[0],action[1])
		x+=action[0]
		y+=action[1]
	elif action_success == "SC":
		x+=action[0]
		y+=action[1]
	elif action_success == "VL":
		action = rotate_left(action[0],action[1])
		x+=action[0]
		y+=action[1]
	if(is_outside_boundary_or_found_obstacle(x,y)):
		x=orig_x
		y=orig_y
	#print (x,y)
        return (x,y,get_reward(x,y,t),get_actual_reward(x,y,t),grid_rules.get((x,y))=="GOAL")


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
apply_optimal_policy = False
td_errors=[]
def get_state():
	global s
	s_idx = randint(0,22)
	return s[s_idx]
	

def plot_error(error,time_steps,xlabel,ylabel,title,filename,y_error=[]):
        global alphas
           #new_alphas = [np.log10(x) for x in alphas]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
	plt.ylim(-10,5)
        if len(y_error)==0:
                plt.plot(np.array(time_steps),np.array(error))
        else:
                plt.axhline(y=np.max(error),linestyle='--',label="max: "+str(np.max(error)))
                plt.errorbar(np.array(time_steps),np.array(error), yerr=y_error,fmt='-', color='black', ecolor='lightgray',label="rewards vs episodes")
        #plt.show()
        plt.legend()
        plt.savefig(filename)
        plt.clf()

def update_weights(exp_tuples,episode_rewards,alpha,beta):
	global q_value_function
        global e_traces
        global e_value
        global value_function
	lent = len(episode_rewards)
	total_rewards=0.0
	gt_values= []
	
	for idx in range(lent-1,-1,-1):
		total_rewards= gamma*total_rewards + episode_rewards[idx]
		gt_values.append(total_rewards)
	gt_values.reverse()
	for idx in range(0,lent):
		experience = exp_tuples[idx]
		prev_x,prev_y,action_idx,x,y,actual_reward = experience
		old_q_index = (prev_x*grid_len+prev_y)*actions_len + action_idx
		target = gt_values[idx]
                error = target #- value_function[prev_x*grid_len+prev_y]
                mask = np.zeros(e_traces.shape,dtype=bool)
                mask[(prev_x*grid_len+prev_y)*actions_len:(prev_x*grid_len+prev_y)*actions_len+4]=1
                q_value_function[mask]+= alpha*error*(-1.0*s_a_probs[mask])
		q_value_function[old_q_index]+=1*alpha*error
		#value_function[prev_x*grid_len+prev_y] += alpha*error
	print("Norm: " + str(np.sum(q_value_function*q_value_function)))
        update_probabilities(epsilon)

def run_episode(alpha,beta,update_value_function=False,req_trials=100,epsilon=0.1):
	trials = 0
	start_x=0
	global q_value_function
	global e_traces
	global e_value
	global value_function
	start_y=0
	reward_sum=0.0
	td_errors = []
	trial_rewards=[]
	
	while trials <= req_trials:
			e_traces = np.array([0.0 for x in range(0,100)])
			e_value = np.array([0.0 for x in range(0,25)])
			#import ipdb; ipdb.set_trace()
                        REWARD = ( 0.0)
                        flag=False
			state = 0#get_state()
                        x=state/grid_len
                        y=state%grid_len
                        time_steps=0
                        #print ("Time step: " + str( time_steps)+"\n"),
                        #print_timestep(x,y),
                        REWARD = 0.0
                        prev_x = x
                        prev_y = y
			td_error = 0.0
			action_idx,success_idx = select_action(state)
			episode_rewards = []
			exp_tuples = []
                        while(flag==False and time_steps<1000):
                                x,y,new_reward,actual_reward,flag = get_new_state(x,y,action_idx,success_idx,time_steps)
				REWARD += actual_reward
				exp_tuples.append((prev_x,prev_y,action_idx,x,y,new_reward))
				action_idx,success_idx = select_action(x*grid_len+y)
				prev_x = x
				prev_y = y
				episode_rewards.append(new_reward)	
					
                                time_steps+=1
			update_weights(exp_tuples,episode_rewards,alpha,beta)
                   	trial_rewards.append(REWARD)
			print("Reward for episode: "+ str(trials) + " " +str(REWARD)+"\n")
                        trials+=1
			#print trials
				
        trial_rewards = np.array(trial_rewards)
	if update_value_function==False:
		print("alpha: " + str(np.log10(alpha)))
		print("epsilon: " + str(epsilon))
        	print("Minimum discounted reward: " + str(np.min(trial_rewards)))
        	print("Maximum discounted reward: " + str(np.max(trial_rewards)))
	
        	print("Mean discounted reward discounted reward: " + str(np.mean(trial_rewards)))
        	print("Standard Deviation of discounted rewards: " + str(np.std(trial_rewards)))
                	#trial_rewards = x / np.linalg.norm(trial_rewards)
                	#plot_histogram('Discounted Rewards','Frequency','Histogram of Discounted Rewards',trial_rewards,'DiscountedRewardsHistogram.png')
                	#plot_histogram('Discounted Rewards on using Optimal Policy','Frequency','Histogram of Discounted Rewards on using Optimal Policy',trial_rewards,'DiscountedRewardsHistogramOptimal.png')
        bellman_error = np.mean(td_errors)
	errors = [np.power(x,2) for x in td_errors]
	#if update_value_function:	
		#plot_error(trial_rewards,[i for i in range(0,len(trial_rewards))],"trials","rewards","rewards vs trials","rewards_gridworld_"+str(alpha)+".png")
		#plot_error(errors,[i for i in range(0,len(td_errors))],"time_steps","td_errors","td error vs time_steps","td_error_gridworld_" + str(alpha) + ".png")
	return trial_rewards

if __name__=='__main__':
	
	trials = 100
        if len(sys.argv)>1:
                trials = max(2,int(sys.argv[1]) )	
	global epsilon
	print epsilon
	global ld
	global ldv
	lds = [0.1,0.3,0.5,0.7]
	ldvs = [0.2,0.4,0.6,0.8]
	lent = len(alphas)
	alpha_cnt =0
	alpha_rewards = {}
	epsilon_values = [10,20,5,0.1,0.5,0.2]
	for x in epsilon_values:
		alpha_rewards[x]={}
	gridvalues = [[1,0.1,0.2],[1,0.01,0.2],[1,0.001,0.2],[1,0.05,0.2]]
	
	#for new_ld in lds:
		#for new_ldv in ldvs:
	for value in gridvalues:
		    	ld = 0.2
		    	ldv = 0.2
		    #for eps in epsilon_values:
		    #alpha_cnt=0
		    #  while alpha_cnt < lent:
			trial_cnt=0
			#value = [eps,alphas[alpha_cnt]]
			epsilon = value[0]

			total_rewards=[]
			while trial_cnt< trials:	
				initialize_q_value_function()
				initialize_softmax_probs()		
				#initialize_epsilon_greedy_probs
				mean_reward = run_episode(value[1],value[1],True,2000,value[0])
				print "Reward: " + str(np.max(mean_reward))
				total_rewards.append(mean_reward)
				trial_cnt+=1
				
			trial_rewards = np.mean(total_rewards,axis=0)
			std = np.std(total_rewards,axis=0)
			plot_error(trial_rewards,[i for i in range(0,len(trial_rewards))],"episodes","rewards","rewards vs episodes","reinforce_gridworld_rewards_"  +str(value[0])+ "_"+ str(value[1])+"_" + str(ld) + "_"+ str(ldv)+".png",std)
			alpha_cnt+=1
	 
