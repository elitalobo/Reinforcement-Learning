import os
import sys
filename = sys.argv[1]
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
file2 = open(sys.argv[2],'r')
file3 = open(sys.argv[3],'r')
finalname = sys.argv[4]
filet = open(filename,'r')
cnt=0
z=[]
b = set([])
for x in filet:
	cnt+=1
	print cnt
	x = x.strip()
	x=x.strip("\n")
	y = x.split(",")
	#if(len(y)%10==1):
	#	y=y[1:]
	b.add(len(y))
	print b
	r = np.array([float(k) for k in y])
	z.append(r)
z=np.array(z)
print("done converting z")	
mean = np.mean(z,axis=0)
std = np.std(z,axis=0)
print("got mean and std")
	
#mean_f = open("mean.txt",'r')
#std_f = open("std.txt",'r')
#mean = []
'''std=[]
for x in mean_f:
        x=x.strip(" ")
        x=x.strip("\n")
        x = x.split(",")
        mean = np.array([float(y) for y in x])

print("got mean")
for y in std_f:
        y=y.strip(" ")
        y=y.strip("\n")
        y = y.split(",")
        std = np.array([float(x) for x in y])

print("got std")
'''
x=[i for i in range(0,len(y))]

print("got x")
mean1 =[]
for value in file2:
	value = value.strip("\n").strip()
	value = float(value)
	mean1.append(value)
x1 = [i for i in range(0,len(mean1))]

mean2 =[]
for value in file3:
        value = value.strip("\n").strip()
        value = float(value)
        mean2.append(value)
x2 = [i for i in range(0,len(mean2))]
	
x1 = np.array(x1)
x2 = np.array(x2)
mean1 = np.array(mean1)
mean2 = np.array(mean2)

plt.figure()
axes = plt.gca()
ymin = np.min(mean) - np.max(std)
ymax = np.max(mean) + np.max(std)
print ymin
print ymax
print ("got min and max")
xmin =0
xmax = len(mean)
print mean2
axes.set_xlim([0,150])
axes.set_ylim([-4,5])
plt.title(finalname)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,mean,'-',color='paleturquoise', label="CE")
plt.plot(x2,mean2, label="Qlearning")
plt.plot(x1,mean1,label="Sarsa")
plt.legend()
plt.savefig(finalname+".png")
plt.clf()

print ("plotted error bars")


