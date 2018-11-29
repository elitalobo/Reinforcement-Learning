import json
import sys
import numpy as np
from numpy.random import randint
import operator
import matplotlib.pylab as plt
import random
import math
gamma=1
order = 3
actions_len=2
epsilons = [0.3,0.15,0.075,0.05]

action_index = {-10:0,10:1}
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

def softmax(x,episode_cnt,epsilon):
    sigma = epsilon
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))*sigma)
    return (e_x) / e_x.sum()


def pick_action():
        action_picked = random_pick(actions,action_probabilities)
        return action_picked[0]

def pick_action_success():
        action_success = random_pick(action_result,action_result_probabilities)
        return action_success

#def softmax(x):
#    sigma = 0.5
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp((x - np.max(x))*sigma)
#    return (e_x) / e_x.sum()

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
'''
epsilon=0.3
def e_greedy(x,episode_cnt,eps):
        global epsilon
        #if(episode_cnt<100):
        #       epsilon = epsilons[int(episode_cnt/(100/len(epsilons)))]
        if episode_cnt<25:
                epsilon = 0.3
        elif episode_cnt<50:
                epsilon = 0.15
        elif episode_cnt<75:
                epsilon = 0.075
        elif episode_cnt < 95:
                epsilon = 0.030
        elif episode_cnt < 100:
                epsilon = 0.005
        else:
                epsilon = 0.000001
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

'''

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



forces =[-10.0,10.0]
force_probabilities = [0.5,0.5]
def random_pick(some_list, probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
                cumulative_probability += item_probability
                if x < cumulative_probability: break
        return item,item_probability

delta=0.02
time_limit = 20+10*delta
max_fail_ang= math.pi/2
min_fail_ang=(-1.0*math.pi/2)
force = 10.0
gravitational_constant=g=9.8
cart_mass=1.0
pole_mass=0.1
pole_length=0.5 #half_length=0.5
max_dist = 3.0
min_dist=-3.0
max_vel = 10 #10.0
min_vel= -10 #-10.0
max_ang_vel = math.pi
min_ang_vel = -1.0*math.pi
min_force = -10.0
max_force = 10.0
def linear(X):
	a=1
	linear_f = lambda x: a*x
        vectorized_linear = np.vectorize(linear_f)
        res = vectorized_linear(X)
        return res

def sigmoid(X):
	sigmoid_f = lambda x: 1.0/(1.0+np.exp(-x))
        vectorized_sigmoid = np.vectorize(sigmoid_f)
        res = vectorized_sigmoid(X)
	return res

def sigmoid_final(x):
	return (1.0/(1.0+np.exp(-x)));


def tanh(X):

	tanh_f = lambda x: (2.0/(1+np.exp(-2*x)))-1
        vectorized_tanh = np.vectorize(tanh_f)
        res = vectorized_tanh(X)
        return res
	

def relu(X):
	maxm_f = lambda x:  max(0.0,x)
        vectorized_maxm = np.vectorize(maxm_f)
        res = vectorized_maxm(X)
        return res

def leaky_relu(X):
	leaky_relu_f = lambda x: x if x>=0 else 0.01*x
        vectorized_leaky_relu = np.vectorize(leaky_relu_f)
        res = vectorized_leaky_relu(X)
        return res

def parametrized_relu(X,a):
	leaky_relu_f = lambda x: x if x>=0 else a*x
        vectorized_leaky_relu = np.vectorize(leaky_relu_f)
        res = vectorized_leaky_relu(X)
        return res

'''def softmax(arr):
	res = np.array(arr)
	maxm= np.max(res)
	res=res-maxm
	exp_f = lambda x: np.exp(x)
	# Create a vectorized function
	vectorized_exp = np.vectorize(exp_f)
	res = vectorized_exp(res) 
	res = (res*1.0)/np.sum(res)
	return res
'''	


from math import sin, cos
class CartPole() :
	def __init__(self,dist,vel,ang_vel,ang,order,alpha):
		self.episodes = 0				
		self.alpha = alpha
		self.result=0.0
		self.dist=dist;
		self.vel = vel;
		self.ang_vel = ang_vel;
		self.ang = ang;
		self.old_acc=0.0;
		self_acc=0.0;
		self.old_x=0.0;
		self.old_vel=0.0;
		self.old_ang=0.0;
		self.old_ang_vel=0.0; 
		self.old_dist=0.0
		self.params =[0.0,0.0,0.0,0.0,0.0]
		self.old_ang_acc=0.0
		self.ang_acc=0.0
		self.cache={}
		self.no_of_inputs=4
		self.no_of_hidden_nodes=3
		self.time_steps=0
		self.weights1=None
		self.weights2=None
		self.biases=None
		self.forces = [-10,10]
		self.force =0.0
		self.old_action = 0.0
		self.action = 0.0
		self.order = order
		self.epsilon=0.3
		#self.value_weights = np.array([np.random.normal(loc =0, scale =1, size=(1,np.power(self.order+1,4))),np.random.normal(loc =0, scale =1, size=(1,np.power(self.order+1,4)))])
	 	self.value_weights = np.array([np.zeros((1,int(np.power((self.order+1),4)))),np.zeros((1,int(np.power((self.order+1),4))))])
		self.fourier_vector = np.array([])
		self.initialize_fourier_vector()
	def update_angular_acceleration(self):
		global g
		global pole_mass
		global cart_mass
		global pole_length
		ang_acc = (g*sin(self.ang) + (cos(self.ang)*((-self.force-(pole_mass*pole_length*pow(self.ang_vel,2)*sin(self.ang)))/(cart_mass+pole_mass))))/(pole_length*((4.0/3)-((pole_mass*pow(cos(self.ang),2))/(cart_mass+pole_mass))))
		self.old_ang_acc=self.ang_acc;	
		self.ang_acc = ang_acc

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
				k=0
				l=0
				while k <= lent:
					l=0
					while l <= lent:
							vec = np.array([i,j,k,l])
							if self.fourier_vector.size==0:
								self.fourier_vector = vec
							else:
								self.fourier_vector = np.vstack((self.fourier_vector,vec))
							l+=1
					k+=1
				j+=1
			i+=1
	def apply_cos(self,arr):
		#res = [cos(y) for y in arr]
		res = np.sum(arr)			
		return np.cos(res*math.pi)	

	def evaluate_fourier(self,dist,ang,vel,ang_vel):
		dist,ang,vel,ang_vel = self.get_normalized(dist,ang,vel,ang_vel)
		req = np.array([dist,ang,vel,ang_vel])
		evaluated_fourier = np.cos(np.matmul(self.fourier_vector,req)*np.pi)
		#res = np.apply_along_axis(self.apply_cos, 1, evaluated_fourier)	
		res = evaluated_fourier.reshape(evaluated_fourier.shape[0],1)
		return res	
	def evaluate_value_estimate(self,Reward,new_action,flag,update=False):
		evaluated_fourier = self.evaluate_fourier(self.old_dist,self.old_ang,self.old_vel,self.old_ang_vel)
		target = Reward
		if flag==False:
			target = Reward + gamma*(np.matmul(self.value_weights[action_index[new_action]],self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel)))
		td = ( target - np.matmul(self.value_weights[action_index[self.force]],evaluated_fourier))
		#print td
		#import ipdb; ipdb.set_trace()	
		#print "start"
		#print self.value_weights.shape
		#print evaluated_fourier
		#print np.matmul(self.value_weights,np.transpose(evaluated_fourier))
		#print np.matmul(self.value_weights,np.transpose(self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel,self.force)))
		#print "end"
		#print td
		#assert(alpha==1.0)
		if update:
			self.value_weights[action_index[self.force]] = self.value_weights[action_index[self.force]] + self.alpha*td*np.transpose(evaluated_fourier)
			#print self.value_weights
			#k = self.alpha*td*evaluated_fourier
			#if np.isnan(k).any():
			#	import ipdb; ipdb.set_trace()	
		#else:
			#print "not updating weights"
		return td

	def update_angular_velocity(self):
		global max_ang_vel
		global min_ang_vel
		self.old_ang_vel = self.ang_vel;
		self.ang_vel += (delta*self.ang_acc)
		if(self.ang_vel< min_ang_vel):
			self.ang_vel = min_ang_vel
		if(self.ang_vel > max_ang_vel):
			self.ang_vel = max_ang_vel
	
	def update_angle(self):
		global max_ang
		global min_ang
		self.old_ang = self.ang;
		self.ang += (delta*self.ang_vel)
		flag = False
		if(self.ang >= max_fail_ang):
			self.ang = max_fail_ang
			flag = True
		if(self.ang <=min_fail_ang):
			self.ang = min_fail_ang
			flag = True
		return flag
	
	def update_action(self,force):
		self.old_action = self.action
		self.action = force
	
	def update_velocity(self):
		global max_vel
		global min_vel
		self.old_vel = self.vel
		self.vel +=(self.acc*delta)
		if(self.vel > max_vel):
			self.vel = max_vel
		if(self.vel < min_vel):
			self.vel = min_vel 

	def update_acceleration(self):
		global g
		global pole_length
		global cart_mass
		global pole_mass
		acc = (self.force+(pole_mass*pole_length)*((pow(self.ang_vel,2)*sin(self.ang))-(self.ang_vel*cos(self.ang))))/(cart_mass+pole_mass)
		self.old_acc = self.acc
		self.acc = acc
	
	
	def update_distance(self):
		global max_dist
		global min_dist
		self.old_dist = self.dist
		self.dist += (delta*self.vel)
		flag = False
		if(self.dist >= max_dist):
			self.dist = max_dist
			flag = True
		if( self.dist <= min_dist):
			self.dist = min_dist
			flag = True
		return flag
		
	def reset_state(self):
		self.dist=0.0
		self.old_dist=0.0
		self.acc=0.0
		self.old_acc=0.0
		self.ang_acc=0.0
		self.old_ang_acc=0.0
		self.ang_vel=0.0
		self.old_ang_vel=0.0
		self.vel=0.0
		self.old_vel=0.0
		self.ang=0.0
		self.old_ang=0.0
		self.dist=0.0
		self.old_dist=0.0
		self.old_action = 0.0
		self.action = 0.0
		self.force = 0.0



	def get_action(self):
		import ipdb;
                force_values = [np.matmul(self.value_weights[0],self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel))[0][0],np.matmul(self.value_weights[1],self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel))[0][0]]
                force_probabilities = softmax(force_values,self.episodes,self.epsilon)
                force, proba = random_pick(forces,force_probabilities)
		return force
	
	def update_state(self,force):
		flag=False
		self.force = force
		self.update_angular_acceleration()
		self.update_acceleration()
		self.update_angular_velocity()
		flag = flag or self.update_angle()
		#self.update_acceleration()
		self.update_velocity()
		flag = flag or self.update_distance()
		self.update_action(force)
		return flag
		
	def get_normalized(self,dist,ang,vel,ang_vel):
		global min_dist
		global max_dist
		global min_vel
		global max_vel
		global min_fail_ang
		global max_fail_ang
		global min_ang_vel
		global max_ang_vel
		global min_force
		global max_force
		dist = (dist-min_dist)/(max_dist-min_dist)
		ang = (ang-min_fail_ang)/(max_fail_ang-min_fail_ang)
		vel = (vel-min_vel)/(max_vel-min_vel)
		ang_vel = (ang_vel-min_ang_vel)/(max_ang_vel-min_ang_vel)
		if math.isnan(dist) or math.isnan(vel) or math.isnan(ang) or math.isnan(ang_vel):
			print "FOUND NAN"
		return dist,ang,vel,ang_vel
	
	def run_episode(self,td_errors,reset=True,update=False,debug=False):
		global time_limit
		if reset:
			self.reset_state()
		flag=False
		REWARD =0.0
		time_steps=0.0
		#print "episode started"
		action = self.get_action()
		while flag==False and time_steps<= time_limit-delta:
			if debug:
				print("Timestep: " + str(time_steps))
				print "old dist: " + str(self.old_dist) + " dist: "  + str(self.dist)
					
                        	print "old vel:" + str(self.old_vel) + "  vel: " + str(self.vel)
                        	print " old ang: " + str(self.old_ang) + " ang: " +  str(self.ang)
                        	print "old ang_vel: " + str(self.old_ang_vel) + " ang_vel: " + str(self.ang_vel)
                        	print "force: " + str(self.force)
			
			flag = self.update_state(action);
			action = self.get_action()	
			td = self.evaluate_value_estimate(1.0,action,flag,update)	
			#print td
			td_errors.append(td)
			if(flag):
				break
			REWARD+=1.0
			time_steps+=delta
			#print time_steps
		print time_steps
		self.episodes+=1
		#if(update):
		#	self.epsilon*=0.99
		print "episode ended"
		if(REWARD > 1010):
			import ipdb; ipdb.set_trace()
		return td_errors,REWARD		


def run_trials(cartpole,trials=1, update=False):
	trial_cnt=0
	td_errors = []
	trial_rewards = []
	while trial_cnt < trials:
		td_errors,reward = cartpole.run_episode(td_errors,True,update,False)
		trial_cnt+=1
		trial_rewards.append(reward)
		print trial_cnt
		print reward
		print "Norm:" +  str(np.sum(cartpole.value_weights[0]*cartpole.value_weights[0]))
	#print len(td_errors)
	#import ipdb; ipdb.set_trace()
	td_errors = [np.power(y,2) for y in td_errors]
	#print "TD errors: "
	#print td_errors
	#print "DONE"
	if(update==False):
		print "Mean rewards: " + str(np.mean(trial_rewards))
	#else:
	#	plot_error(trial_rewards,[i for i in range(0,len(trial_rewards))],"episodes","rewards","rewards vs episodes","temp/cartpole_rewards_" + str(cartpole.order) + "_"+ str(cartpole.alpha)+ "_" + str(cartpole.epsilon)+".png")	
        return cartpole,trial_rewards

if __name__=="__main__":
	trials = 100
        if len(sys.argv)>1:
                trials = max(2,int(sys.argv[1]) )
	global order
	#orders = [1,3,5,7,9,11]
	#orders = [8]
	#(4.0,13.0,0.5)
	orders=[3,4,5,6,7]
	total_rewards=[]
	trial_cnt=0
	gridsearch = [[5,0.001],[7,0.001]]
	for values in  gridsearch:
		filet = "cartpole_sarsa_results_"+str(values[0])+"_"+str(values[1])+"_"
		total_rewards=[]
		trial_cnt=0
		while trial_cnt < trials:
			cartpole = CartPole(0.0,0.0,0.0,0.0,values[0],values[1])
			cartpole,trial_reward = run_trials(cartpole,100,True)
			total_rewards.append(trial_reward)
	
			trial_cnt+=1
		
		trial_rewards = np.mean(total_rewards,axis=0)
		std_dev = np.std(total_rewards,axis=0)
		
		np.savetxt(filet+"mean.txt", trial_rewards, delimiter=',')
		np.savetxt(filet+"std.txt",std_dev,delimiter=',')
		print trial_rewards
		plot_error(trial_rewards,[i for i in range(0,len(trial_rewards))],"episodes","rewards","rewards vs episodes","softmax_sarsa_cartpole_rewards_" + str(cartpole.order) + "_"+ str(cartpole.alpha)+ "_" + str(cartpole.epsilon)+".png",std_dev)	
