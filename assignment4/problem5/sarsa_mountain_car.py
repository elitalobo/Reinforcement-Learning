import sys

import numpy as np
from numpy.random import randint
import operator
import matplotlib.pylab as plt
import random
import math
gamma=1
#plt.style.use('seaborn-whitegrid')
order = 3
actions_len=3
epsilons = [0.3,0.15,0.075,0.05]
time_limit=50000
action_index = {-1:0,0:1,1:2}
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

def filter(arr):
        temp = []
        for x in arr:
                if x<-1000:
			temp.append(-1000)
                        continue
                else:
                        temp.append(x)
        return temp


def pick_action():
        action_picked = random_pick(actions,action_probabilities)
        return action_picked[0]

def pick_action_success():
        action_success = random_pick(action_result,action_result_probabilities)
        return action_success

def softmax(x,episode_cnt,epsilon):
    sigma = 0.5
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))*sigma)
    return (e_x) / e_x.sum()

def e_greedy(x,episode_cnt,epsilon):
	#if(episode_cnt<100):
	#	epsilon = epsilons[int(episode_cnt/(100/len(epsilons)))]
	
	probabilities = []
        maxm = np.max(x)
        cnt=0
        max_arr = []
	
        for value in x:
                if value == maxm:
                        max_arr.append(cnt)
                cnt+=1
	try:
        	idx = max_arr[0]
	except:
		import ipdb; ipdb.set_trace()
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

def select_action(s):
        #import ipdb; ipdb.set_trace()
        global s_a_probs
        global actions
        #import ipdb; ipdb.set_trace()
        action_probs = s_a_probs[s*actions_len:s*actions_len+4]
        action_idx = random_pick(actions,action_probs)
        success_idx = random_pick(action_result,action_result_probabilities)
        return action_idx,success_idx

def plot_error(error,time_steps,xlabel,ylabel,title,filename,y_error=[]):
        global alphas
	   #new_alphas = [np.log10(x) for x in alphas]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(y_error)==0:
		plt.plot(np.array(time_steps),np.array(error))
	else:
		plt.axhline(y=np.max(error),linestyle='--',label="max: "+str(np.max(error)))
		plt.errorbar(np.array(time_steps),np.array(error), yerr=y_error,fmt='-', color='black', ecolor='lightgray',label="rewards vs episodes")
        #plt.show()
	plt.legend()
        plt.savefig(filename)
        plt.clf()



action_probabilities = [0.5,0.5]
def random_pick(some_list, probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
                cumulative_probability += item_probability
                if x < cumulative_probability: break
        return item,item_probability

max_dist = 0.5
min_dist=-1.2
max_vel = 0.07 #10.0
min_vel= -0.07 #-10.0


from math import sin, cos
class MountainCar() :
	def __init__(self,order,alpha,epsilon):
		self.episodes = 0				
		self.alpha = alpha
		self.result=0.0
		self.dist=-0.5;
		self.vel = 0.0;
		self.old_vel=0.0;
		self.old_dist=0.0
		self.actions = [-1,0,1]
		self.action =0
		self.old_action = 0.0
		self.action = 0.0
		self.order = order
		self.epsilon=epsilon
		self.value_weights = np.array([np.random.normal(loc =0, scale =1, size=(1,np.power(self.order+1,2))),np.random.normal(loc =0, scale =1, size=(1,np.power(self.order+1,2))),np.random.normal(loc =0, scale =1, size=(1,np.power(self.order+1,2)))])
	 	#self.value_weights = np.array([np.zeros((1,int(np.power((self.order+1),2)))),np.zeros((1,int(np.power((self.order+1),2))))])
		self.fourier_vector = np.array([])
		self.initialize_fourier_vector()
	

	def initialize_fourier_vector(self):
		i=0
		j=0
		k=0
		l=0
		lent = self.order
		while i <= lent:
			j=0
			k=0	
			l=0
			while j <= lent:
				vec = np.array([i,j])
				if self.fourier_vector.size==0:
					self.fourier_vector = vec
				else:
					self.fourier_vector = np.vstack((self.fourier_vector,vec))
				j+=1
			i+=1
	def apply_cos(self,arr):
		#res = [cos(y) for y in arr]
		res = np.sum(arr)			
		return np.cos(res*math.pi)	

	def evaluate_fourier(self,dist,vel):
		dist,vel = self.get_normalized(dist,vel)
		req = np.array([dist,vel])
		evaluated_fourier = np.cos(np.matmul(self.fourier_vector,req)*np.pi)
		#res = np.apply_along_axis(self.apply_cos, 1, evaluated_fourier)	
		res = evaluated_fourier.reshape(evaluated_fourier.shape[0],1)
		return res	
	def evaluate_value_estimate(self,Reward,new_action,flag,update=False):
		evaluated_fourier = self.evaluate_fourier(self.old_dist,self.old_vel)
		target = Reward
		if flag==False:
			target = Reward + gamma*(np.matmul(self.value_weights[action_index[new_action]],self.evaluate_fourier(self.dist,self.vel)))
		td = ( target - np.matmul(self.value_weights[action_index[self.action]],evaluated_fourier))
		#print td
		#import ipdb; ipdb.set_trace()	
		#print "start"
		#print self.value_weights.shape
		#print evaluated_fourier
		#print np.matmul(self.value_weights,np.transpose(evaluated_fourier))
		#print np.matmul(self.value_weights,np.transpose(self.evaluate_fourier(self.dist,self.vel)))
		#print "end"
		#print td
		#assert(alpha==1.0)
		if update:
			self.value_weights[action_index[self.action]] = self.value_weights[action_index[self.action]] + self.alpha*td*np.transpose(evaluated_fourier)
		return td

	
	def update_action(self,action):
		self.old_action = self.action
		self.action = action
	
	def update_velocity(self):
		global max_vel
		global min_vel
		self.old_vel = self.vel
		self.vel += 0.001*self.action-(0.0025)*cos(3*self.dist)
		if(self.vel < min_vel):
			self.vel = min_vel
		if(self.vel > max_vel):
			self.vel = max_vel
		 
	
	def update_distance(self):
		global max_dist
		global min_dist
		self.old_dist = self.dist
		self.dist += (self.vel)
		flag = False
		if(self.dist >= max_dist):
			self.dist = max_dist
			self.vel = 0.0 
			flag = True
		if( self.dist <= min_dist):
			self.dist = min_dist
			self.vel = 0.0
		return flag
		
	def reset_state(self):
		self.dist=-0.5
		self.old_dist=0.0
		self.vel=0.0
		self.old_vel=0.0
		self.old_action = 0.0
		self.action = 0



	def get_action(self):
		import ipdb;
                action_values = [np.matmul(self.value_weights[0],self.evaluate_fourier(self.dist,self.vel))[0][0],np.matmul(self.value_weights[1],self.evaluate_fourier(self.dist,self.vel))[0][0],np.matmul(self.value_weights[2],self.evaluate_fourier(self.dist,self.vel))[0][0]]
                action_probabilities = softmax(action_values,self.episodes,self.epsilon)
                action, proba = random_pick(self.actions,action_probabilities)
		return action
	
	def update_state(self,action):
		flag=False
		self.update_action(action)
		self.update_velocity()
		flag = flag or self.update_distance()
		return flag
		
	def get_normalized(self,dist,vel):
		global min_dist
		global max_dist
		global min_vel
		global max_vel
		dist = (dist-min_dist)/(max_dist-min_dist)
		vel = (vel-min_vel)/(max_vel-min_vel)
		if math.isnan(dist) or math.isnan(vel):
			print "FOUND NAN"
		return dist,vel
	
	def run_episode(self,td_errors,reset=True,update=False,debug=False):
		global time_limit
		if reset:
			self.reset_state()
		flag=False
		delta=1
		REWARD =0.0
		time_steps=0.0
		#print "episode started"
		action = self.get_action()
		while flag==False and time_steps<time_limit:
			if debug:
				print("Timestep: " + str(time_steps))
				print "old dist: " + str(self.old_dist) + " dist: "  + str(self.dist)
					
                        	print "old vel:" + str(self.old_vel) + "  vel: " + str(self.vel)
                        	print " old ang: " + str(self.old_ang) + " ang: " +  str(self.ang)
                        	print "old ang_vel: " + str(self.old_ang_vel) + " ang_vel: " + str(self.ang_vel)
                        	print "action: " + str(self.action)
			
			flag = self.update_state(action)
			action = self.get_action()	
			td = self.evaluate_value_estimate(-1.0,action,flag,update)	
			#print td
			td_errors.append(td)
			if(flag):
				break
			REWARD+=-1.0
			time_steps+=1
			#print time_steps
		print "REWARD:" + str(REWARD)
		self.episodes+=1
		#self.epsilon*=0.90
		print "episode ended"
		return td_errors,REWARD	


def run_trials(mountain_car,trials=1, update=False):
	trial_cnt=0
	td_errors = []
	trial_rewards = []
	while trial_cnt < trials:
		td_errors,reward = mountain_car.run_episode(td_errors,True,update,False)
		trial_cnt+=1
		trial_rewards.append(reward)
		print trial_cnt
		print reward
		print "Norm:" +  str(np.sum(mountain_car.value_weights[0]*mountain_car.value_weights[0]))
                print "Norm:" +  str(np.sum(mountain_car.value_weights[1]*mountain_car.value_weights[1]))
                print "Norm:" +  str(np.sum(mountain_car.value_weights[2]*mountain_car.value_weights[2]))
	#print len(td_errors)
	#import ipdb; ipdb.set_trace()
	td_errors = [np.power(y,2) for y in td_errors]
	#print "TD errors: "
	#print td_errors
	#print "DONE"
	if(update==False):
		print "Mean rewards: " + str(np.mean(trial_rewards))
	else:
		rewards = filter(trial_rewards)	
		#plot_error(rewards,[i for i in range(0,len(rewards))],"episodes","rewards","rewards vs episodes","final/mountain_car_rewards_" + str(mountain_car.order) + "_"+ str(mountain_car.alpha) + "_" + str(mountain_car.epsilon) +".png")	
        return mountain_car,rewards

if __name__=="__main__":
	trials = 100
        if len(sys.argv)>1:
                trials = max(2,int(sys.argv[1]) )	
	global order
	orders = [1,3,5,7,9,11]
	orders = [2,3,4]
	#epsilons = [0.1,,0.01,0.05,0.001,0.005]
	epsilons = [0.005,0.001,0.0001,0.01,0.05]
	total_rewards=[]
	#[4,0.0056,0.05],[2,0.001,0.01]
	gridsearch = [[2,0.001778,0.0007],[5,0.001778,0.0001],[3,0.00316,0.0001]]
	#for order in orders:
	#	alphas = [np.power(10.0,-1.0*x) for x in np.arange(2.5,4.0,0.5)]
	#	for alpha in alphas:
	#	   for epsilon in epsilons:
	for values in gridsearch:
			total_rewards=[]
			trial_cnt=0
			while trial_cnt < trials:
                                #mountain_car = MountainCar(order,alpha,epsilon)
				mountain_car = MountainCar(values[0],values[1],values[2])
				mountain_car,trial_reward = run_trials(mountain_car,1000,True)
		
				total_rewards.append(trial_reward)
                        	trial_cnt+=1
			std_dev = np.std(total_rewards,axis=0)
               		rewards = np.mean(total_rewards,axis=0)
                	print rewards
                	plot_error(rewards,[i for i in range(0,len(rewards))],"episodes","rewards","rewards vs episodes","sarsa_softmax_mountain_car_rewards_" + str(mountain_car.order) + "_"+ str(mountain_car.alpha)+ "_" + str(mountain_car.epsilon)+".png",std_dev)

