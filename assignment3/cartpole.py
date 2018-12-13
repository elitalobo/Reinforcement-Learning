import sys
import numpy as np

import operator
import matplotlib.pylab as plt
import random

import math
gamma=1
order = 3

def plot_error(errors,alphas):

        global order
        alphas = [np.log10(x) for x in alphas]
        plt.title("TD error vs alpha")
        plt.xlabel("alpha")
        plt.ylabel("TD error")
        x_min = np.min(alphas)
        x_max = np.max(alphas)
        #y_min = np.min(error)
        #y_max = np.max(error)
        plt.xlim(-10,0)
        #plt.ylim(y_min-2,y_max+2)
	orders=[3,5]
	i=0
        for error in errors:
		if orders[i]==3:
			error = error[1:]
			alphas = alphas[1:]
		print error
		print alphas
		if orders[i]==5:
			error = error[2:]
			alphas = alphas[1:]
		
                plt.plot(np.array(alphas),np.array(error),label="order "+str(orders[i]))
		i=i+1
        plt.legend()
        plt.savefig("TD_error_cartpole.png")


forces =[10.0,-10.0]
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
	def __init__(self,dist,vel,ang_vel,ang,order,alpha):
				
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
		self.order = order
	 	self.value_weights = np.zeros((1,int(np.power((self.order+1),4))))
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
	def evaluate_value_estimate(self,Reward,flag,update=False):
		evaluated_fourier = self.evaluate_fourier(self.old_dist,self.old_ang,self.old_vel,self.old_ang_vel)
		target = Reward
		if flag==False:
			target = Reward + gamma*(np.matmul(self.value_weights,self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel)))
		td = ( target- np.matmul(self.value_weights,evaluated_fourier))
		#print td
		#import ipdb; ipdb.set_trace()	
		#print "start"
		#print self.value_weights.shape
		#print evaluated_fourier
		#print np.matmul(self.value_weights,np.transpose(evaluated_fourier))
		#print np.matmul(self.value_weights,np.transpose(self.evaluate_fourier(self.dist,self.ang,self.vel,self.ang_vel)))
		#print "end"
		#print td
		#assert(alpha==1.0)
		if update:	
			self.value_weights = self.value_weights + self.alpha*td*np.transpose(evaluated_fourier)
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
		#if(self.ang_vel< min_ang_vel):
		#	self.ang_vel = min_ang_vel
		#if(self.ang_vel > max_ang_vel):
		#	self.ang_vel = max_ang_vel
	
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
	
	def update_velocity(self):
		global max_vel
		global min_vel
		self.old_vel = self.vel
		self.vel +=(self.acc*delta)
		#if(self.vel > max_vel):
		#	self.vel = max_vel
		#if(self.vel < min_vel):
		#	self.vel = min_vel 

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
			
	def get_force(self):
		force, proba = random_pick(forces,force_probabilities)
		return force
	
	def update_state(self):
		self.force = self.get_force()
		flag=False
		self.update_angular_acceleration()
		self.update_acceleration()
		self.update_angular_velocity()
		flag = flag or self.update_angle()
		#self.update_acceleration()
		self.update_velocity()
		flag = flag or self.update_distance()
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
			td = self.evaluate_value_estimate(1.0,flag,update)	
			#print td
			
			td_errors.append(td)
			REWARD+=1.0	
			if(flag):
				break
			#REWARD+=1.0
			time_steps+=delta
			#print time_steps
		#print time_steps
		#print "episode ended"
		return td_errors	
	


def run_trials(cartpole,trials=1, update=False):
	trial_cnt=0
	td_errors = []
	while trial_cnt < trials:
		td_errors = cartpole.run_episode(td_errors,True,update)
		trial_cnt+=1
		#print trial_cnt
	#print len(td_errors)
	#import ipdb; ipdb.set_trace()
	td_errors = [np.power(y,2) for y in td_errors]
	#print "TD errors: "
	#print td_errors
	#print "DONE"
        return cartpole,np.mean(np.array(td_errors))
if __name__=="__main__":
	global order
	orders = [3,5]
	errors = []
	for order in orders:
		alphas = [np.power(10.0,-1.0*x) for x in np.arange(1,10,1)]
		td_errors = []
		for alpha in alphas:
			cartpole = CartPole(0.0,0.0,0.0,0.0,order,alpha)
			cartpole,td_error = run_trials(cartpole,1000,True)
			print "value estimate:"
			print np.matmul(cartpole.value_weights,cartpole.evaluate_fourier(0.0,0.0,0.0,0.0))
			print "td_error"
			print td_error	
			#print np.matmul(cartpole.value_weights,cartpole.evaluate_fourier(0.0,0.0,0.0,0.0))
			cartpole,td_error = run_trials(cartpole,100,False)
			#print td_error
			td_errors.append(td_error)
	#print td_errors
	#import ipdb; ipdb.set_trace()
			print "value estimate: "
			print np.matmul(cartpole.value_weights,cartpole.evaluate_fourier(0.0,0.0,0.0,0.0))
		print "td errors: "
		print td_errors	
		errors.append(td_errors)
	plot_error(errors,alphas)
	
