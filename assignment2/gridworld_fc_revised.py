import sys
import dill
dill.settings['recurse'] = True
import multiprocessing as mp
import operator
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
	item_cnt=0
	N= len(some_list)
	while item_cnt < N:
		cumulative_probability += probabilities[item_cnt]
		if x < cumulative_probability: break
		item_cnt+=1
	return some_list[item_cnt],probabilities[item_cnt]
	

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

max_timesteps =  1000
num=0
print("starting")
exp_res = [np.power(gamma,num) for num in range(0,max_timesteps+100)]
print("evaluated")

def is_outside_boundary_or_found_obstacle(x,y):
	if(grid_rules.get((x,y))=="INF" or (x>=grid_len or y>=grid_len or x<0 or y<0)):
		return True
	return False

def compute(s_a_probs):
	cnt=0
	total=0.0
	N = len(s_a_probs)
	results = []
	temp=[]	
	sigma=5.0
	while cnt < N:
		if(cnt%4==0 and cnt!=0):
			req_array = np.array(temp)
			req_max = np.max(req_array)
			req_array = [sigma*(np.exp(x-req_max)) for x in req_array]
			req_array= req_array/np.sum(req_array)
			results.append(req_array)
			temp = []
		#req=np.exp(s_a_probs[cnt])
		#total+=req
		temp.append(s_a_probs[cnt])
		
		cnt+=1
	if(len(temp)!=0):
		req_array = np.array(temp)
                req_max = np.max(req_array)
                req_array = [np.exp(x-req_max) for x in req_array]
                req_array= req_array/np.sum(req_array)
                results.append(req_array)	
	assert(len(results[0])==4)
	return results

def pick_action(state,s_a_probs,no_of_actions=4):
	state_id = (state[0]*5)+(state[1])
	action_prob=s_a_probs[state_id]
	assert(len(action_prob)==4)
	assert(len(action_prob)==no_of_actions)
	action_picked = random_pick(actions,action_prob)
	return action_picked[0]
	

def pick_action_success():
	action_success = random_pick(action_result,action_result_probabilities)
	return action_success
	
def get_reward(x,y,t):
	if rewards.get((x,y))==None:
		return 0.0
	else:
		return float(exp_res[t]*rewards[(x,y)]) 

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

def get_new_state(x,y,t,s_a_probs,optimal=False,Action=None,Action_Success=None):
	orig_x = x
	orig_y = y
	if Action == None and optimal==False:
		action = pick_action((x,y),s_a_probs)
	else:
		action = Action
	if optimal:
		action = get_optimal_action(x,y)
	if Action_Success==None:
		action_success = pick_action_success()
	else:
		action_success= Action_Success
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


time_steps=0
trials =1
req_trials = 10000
start_x = 0
start_y=0
trial_rewards = []
import sys
apply_optimal_policy = False


def run_episode(s_a_probs=None):
	REWARD = 0.0
	flag=False
	x=start_x
	y=start_y
	time_steps=0
	#print ("Time step: " + str( time_steps)+"\n"),
	#print_timestep(x,y),
	while(flag==False and time_steps<=1000):
			
		#print("Time step: " + str(time_steps)+"\n")
		x,y,new_reward,flag = get_new_state(x,y,time_steps,s_a_probs,apply_optimal_policy)				
		#print_timestep(x,y)	
		REWARD += new_reward
		#print("reward for this step: " + str(new_reward) + " " + str(time_steps))
		time_steps+=1
	#print("total_reward for trial " + str(trials) + " : " + str(REWARD)),
	return REWARD
	
	#print("Minimum discounted reward: " + str(np.min(trial_rewards)))
	#print("Maximum discounted reward: " + str(np.max(trial_rewards)))

        #print("Mean discounted reward discounted reward: " + str(np.mean(trial_rewards)))
        #print("Standard Deviation of discounted rewards: " + str(np.std(trial_rewards)))
	#trial_rewards = x / np.linalg.norm(trial_rewards)
	#plot_histogram('Discounted Rewards','Frequency','Histogram of Discounted Rewards',trial_rewards,'DiscountedRewardsHistogram.png')
	#plot_histogram('Discounted Rewards on using Optimal Policy','Frequency','Histogram of Discounted Rewards on using Optimal Policy',trial_rewards,'DiscountedRewardsHistogramOptimal.png')


import random
def initialize_mean_cov(N,sigma=15.0):
	#sigma= random.uniform(0,1)
	sigma=0.2  #2.0 1.0
	#mean_vector = [random.uniform(0,1) for x in range(1,N)]
	mean_vector = np.array([random.uniform(0, 1) for x in range(0,N)])
	covariance = (sigma)*np.identity(N) #30,15
	assert(len(covariance)==len(covariance[0]))
	return mean_vector,covariance


def get_mean_covariance(samples,sigma=15):
	eta = 0.5
	K = len(samples)
	N = len(samples[0])
	cov = (sigma)*np.identity(N)
	res = np.zeros((N,N)) 
	mean = np.mean(samples,axis=0)
	for x in samples:
		res +=((x-mean).reshape(1,N)*np.transpose(((x-mean).reshape(1,N)))) 
	
		
	res = (1.0/(K+eta))*(res+cov)
	
	return mean,res
	
	#req_samples= np.array(samples)
	#return np.mean(req_samples, axis=0),np.cov(np.transpose(req_samples))
	
from datetime import datetime	
def fchc(N,sigma=15,trials=12000,n_episodes=80):
	mean_vector, covariance = initialize_mean_cov(N,sigma)
	trial_cnt=0
	initial_mean = mean_vector
	initial_cov = covariance
	
	avg_reward = 0.0
	trial_rewards = []
	current_reward=-100000.0
	while trial_cnt< trials:
		
		k=0
                samples = np.random.multivariate_normal(mean_vector,covariance,size=1)
		sample = samples[0]
		s_a_probs = compute(sample)
		req_samples = [s_a_probs for x in range(0,n_episodes)]
		#pool = mp.Pool(processes=4)
		episode_cnt=0
    		#results = pool.map(run_episode, req_samples)
		#total_returns = sum(results)
		total_returns = 0.0
		#max_return = 0.0
		while episode_cnt < n_episodes:
			Return = run_episode(s_a_probs)
			episode_cnt+=1	
			#max_return = max(max_return,Return)
			total_returns += Return
			trial_rewards.append(Return)
		avg_return=total_returns/n_episodes
		#pool.close()
		#print("Total: " + str(total))
		if(avg_return > current_reward):
			current_reward= avg_return
			mean_vector = sample					
		print("****************	Trial: " + str(trial_cnt) + " is complete**************** " + str(avg_return))
		trial_cnt+=1
	return trial_rewards

import matplotlib.pyplot as plt
from datetime import datetime

if __name__=="__main__":
	new_file = open("gridworld_fc_revised.txt",'w+')
	start = datetime.now()
	trials=int(sys.argv[1]) or 5
	trial_cnt=0
	sigma = 20.5	
	best_reward=0.0
	best_sigma=10.0
	while trial_cnt < trials:
		final_rewards = fchc(25*4,sigma)
		end = datetime.now()
		print (end-start).total_seconds()
		x=[]
		y=[]
		res=str(sigma)
		cnt=0
		for reward in final_rewards:
			x.append(cnt)
			y.append(reward)	
			res+=","
			cnt+=1
			res+= str(reward)
			
		res+="\n"
		new_file.write(res)
		#print y
		#axes = plt.gca()
		#ymin = np.min(y)
		#ymax = np.max(y)
		#xmin =0
		#xmax = len(final_rewards)
		#axes.set_xlim([xmin-1000,xmax])
		#axes.set_ylim([ymin,ymax])
		#plt.scatter(np.array(x),np.array(y))
		#plt.show()
		#plt.savefig("images/grid_world_"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+".png")
		#plt.clf()
		#import ipdb; ipdb.set_trace()	
		trial_cnt+=1
		#sigma+=2	
	new_file.close()

'''

final_rewards=[]
cnt=0
s_a_probs = compute([0.25 for  x in range(0,100)])
while cnt<10000: 
	final_rewards.append(run_episode(s_a_probs))
	cnt+=1

print np.max(np.array(final_rewards))
print np.min(np.array(final_rewards))
print np.mean(np.array(final_rewards))
print np.std(np.array(final_rewards))
'''

