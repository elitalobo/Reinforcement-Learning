import os
import sys
filename = sys.argv[1]
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure()
axes = plt.gca()
ymin = np.min(mean) - np.max(std)
ymax = 5 #np.max(mean) #+ np.max(std)
print ymin
print ymax
print ("got min and max")
xmin =0
xmax = len(mean)
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

plt.errorbar(x,mean,yerr=std,xerr=None,ecolor='black',color='yellow')
#plt.show()
print ("plotted error bars")
plt.savefig(filename.split(",")[0] +".png")
plt.clf()


