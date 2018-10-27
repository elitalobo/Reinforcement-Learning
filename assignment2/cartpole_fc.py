import sys
import numpy as np

import dill
dill.settings['recurse'] = True
import pathos.multiprocessing as mp
import operator
import matplotlib.pylab as plt
import random

import math


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
max_vel = 10.0
min_vel=-10.0
max_ang_vel = math.pi
min_ang_vel = -1.0*math.pi
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

def softmax(arr):
	res = np.array(arr)
	maxm= np.max(res)
	res=res-maxm
	exp_f = lambda x: np.exp(x)
	# Create a vectorized function
	vectorized_exp = np.vectorize(exp_f)
	res = vectorized_exp(res) 
	res = (res*1.0)/np.sum(res)
	return res
	


from math import sin, cos
class CartPole() :
	def __init__(self,dist,vel,ang_vel,ang):
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
		self.params =[0.0,0.0,0.0,0.0]
		self.old_ang_acc=0.0
		self.ang_acc=0.0
		self.cache={}
		self.no_of_inputs=4
		self.no_of_hidden_nodes=3
		self.time_steps=0
		self.weights1=None
		self.weights2=None
		self.biases=None
		self.forces = [10,-10]
		self.force =0.0
	def update_angular_acceleration(self):
		global g
		global pole_mass
		global cart_mass
		global pole_length
		ang_acc = (g*sin(self.ang) + (cos(self.ang)*((-self.force-(pole_mass*pole_length*pow(self.ang_vel,2)*sin(self.ang)))/(cart_mass+pole_mass))))/(pole_length*((4.0/3)-((pole_mass*pow(cos(self.ang),2))/(cart_mass+pole_mass))))
		self.old_ang_acc=self.ang_acc;	
		self.ang_acc = ang_acc

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
		if(self.ang >= max_fail_ang or self.ang <=min_fail_ang):
			return True
		return False
	
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
		if(self.dist >= max_dist or self.dist <= min_dist):
			return True
		return False
		

	def construct_perceptron(self,probabilities):
		lent = len(probabilities)
		self.weights2 = np.array(probabilities[lent-2*self.no_of_hidden_nodes:lent-self.no_of_hidden_nodes]).reshape(self.no_of_hidden_nodes,1)
		self.weights1 = np.array(probabilities[:lent-2*self.no_of_hidden_nodes]).reshape(self.no_of_inputs,self.no_of_hidden_nodes)
		self.biases = np.array(probabilities[lent-self.no_of_hidden_nodes:]).reshape(1,self.no_of_hidden_nodes)
		return self.weights2,self.weights1,self.biases
	def evaluate_perceptron(self):
		inputs = np.array([self.dist,self.vel,self.ang,self.ang_vel]).reshape(1,self.no_of_inputs)
		result = sigmoid(np.matmul(linear(np.add(np.matmul(inputs,self.weights1),self.biases)),self.weights2))
		self.result = result
		return self.result	
	def construct_regression(self,probabilities):
		self.weights1 = probabilities
		return self.weights1,self.weights2,self.biases		
	def evaluate_regression(self):
		inputs = [self.dist,self.vel,self.ang,self.ang_vel]
		cnt=0
		res=0.0
		while cnt < 4:
			res+= inputs[cnt]*self.weights1[cnt]	
			cnt+=1
		return [sigmoid_final(res)]
	
	def evaluate_regression_bool(self):
		inputs = [self.dist,self.vel,self.ang,self.ang_vel]
                cnt=0
                res=0.0
                while cnt < 4:
                        res+= inputs[cnt]*self.weights1[cnt]     
                        cnt+=1
		if res >= 0:
			return [1.0]
		else:
			return [0.0]
	def initialise_perceptron(self,weights1,weights2,biases):
		self.weights2=weights2
		self.weights1=weights1
		self.biases=biases
	
	def reset_state(self):
		self.dist=0.0
		self.old_dist=0.0
		self.acc=0.0
		self.ang_acc=0.0
		self.ang_vel=0.0
		self.old_ang_vel=0.0
		self.vel=0.0
		self.old_vel=0.0
		self.ang=0.0
		self.old_ang=0.0
		self.dist=0.0
		self.old_dist=0.0
	
	def get_force(self):
		#right_probability = self.evaluate_perceptron()[0]
                right_probability = self.evaluate_regression()[0]
                #right_probability = self.evaluate_regression_bool()[0]

		#print "Right probability: " + str(right_probability)
		action_probabilities=[right_probability,1-right_probability]
		action,probability = random_pick(self.forces,action_probabilities) 
		return action

	def update_state(self):
		self.force = self.get_force()
		flag=False
		self.update_angular_acceleration()
		self.update_acceleration()
		self.update_angular_velocity()
		flag = self.update_angle()
		#self.update_acceleration()
		self.update_velocity()
		flag = self.update_distance()
		return flag
		
	def run_episode(self,sample,reset=True,debug=False):
		global time_limit
		if reset:
			self.reset_state()
		flag=False
		REWARD =0.0
		time_steps=0.0
		while flag==False and time_steps<= time_limit-delta:
			if debug:
				print("Timestep: " + str(time_steps))
				print "old dist: " + str(self.old_dist) + " dist: "  + str(self.dist)

                        	print "old vel:" + str(self.old_vel) + "  vel: " + str(self.vel)
                        	print " old ang: " + str(self.old_ang) + " ang: " +  str(self.ang)
                        	print "old ang_vel: " + str(self.old_ang_vel) + " ang_vel: " + str(self.ang_vel)
                        	print "result: " + str(self.result)
                        	print "force: " + str(self.force)

			flag = self.update_state();	
			REWARD+=1.0	
			if(flag):
				break
			#REWARD+=1.0
			time_steps+=delta
		return REWARD
	
	
def initialize_mean_cov(N,timesteps,sigma=15):
        sigma=0.6
	#mean_vector = [random.uniform(0,1) for x in range(1,N)]
	mean_vector = [0.0 for x in range(0,N)]
        mean_vector = np.array([random.uniform(1, 20) for x in range(0,N)])
        covariance = (sigma)*np.identity(N)
        return mean_vector,covariance


def get_mean_covariance(samples,timesteps):
        eta = 0.4
	sigma= 30.0
        #if(timesteps>=20):
        #        sigma=5
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

def fchc(N,sigma=15,trials=200,n_episodes=60):
        mean_vector, covariance = initialize_mean_cov(N,sigma)
        trial_cnt=0
        initial_mean = mean_vector
        initial_cov = covariance

        avg_reward = 0.0
        trial_rewards = []
        current_reward=0.0
        while trial_cnt< trials:

                k=0
                samples = np.random.multivariate_normal(mean_vector,covariance,size=1)
                sample = samples[0]
		cartpole = CartPole(0,0,0,0)
                #s_a_probs = cartpole.(sample)
		weights1,weights2,biases=cartpole.construct_regression(sample)


                req_samples = [sample for x in range(0,n_episodes)]
                #pool = mp.Pool(processes=4)
                episode_cnt=0
                #results = pool.map(cartpole.run_episode, req_samples)
                #total_returns = sum(results)
		total_returns=0.0
                #max_return = 0.0
                #s_a_probs = compute(sample)
                while episode_cnt < n_episodes:
                       	Return = cartpole.run_episode(sample)
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
                print("**************** Trial: " + str(trial_cnt) + " is complete**************** " + str(avg_return))
                trial_cnt+=1
        return trial_rewards

import matplotlib.pyplot as plt
import datetime


if __name__=="__main__":
        new_file = open("cartpole_final.txt",'w+')
        trials=int(sys.argv[1]) or 5
        trial_cnt=0
        sigma = 15.0
        best_reward=0.0
        best_sigma=6
        while trial_cnt < trials:
                final_rewards = fchc(4,sigma)

                x=[]
                y=[]
                res=str(sigma)
                cnt=0
                for reward in final_rewards:
                        x.append(cnt)
                        y.append(reward)
                        res+=","
                        res+= str(reward)
			cnt+=1
                res+="\n"
                new_file.write(res)
                #print y
                #axes = plt.gca()
                #ymin = np.min(y)
                #ymax = np.max(y)
                #xmin =0
                #xmax = len(final_rewards)
                #axes.set_xlim([xmin-10,xmax])
                #axes.set_ylim([ymin,ymax])
                #plt.scatter(np.array(x),np.array(y))
                #plt.show()
                #plt.savefig("cart_pole_fc_hc_"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+".png")
                #plt.clf()
                #import ipdb; ipdb.set_trace()
                trial_cnt+=1
                #sigma+=2
        new_file.close()



