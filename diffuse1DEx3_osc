import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

## This script shows how to solve 1D heat equation numerically.
# Here we have oscillating temperature in the middle

## initialise parameters of heat equation
depth = 10.
timeTotal = 5.
K = 1 # diffusion coefficient
period = 1. # heat oscillation period

## specify parameter for numerical solution
Nz = 400. # number of discrete step in space
dz = depth/Nz # size of discrete step in space
# array of each location in our discrete space
dt = 0.9*dz**2/2/K # size of discrete step in time, note that dt<=dz**2/2 for stability of numerical solution
Z_range = np.arange(0,depth+dz,dz)
t_range = np.arange(0, timeTotal, dt)
Nt = int(timeTotal/dt) # total number of discrete time points
midpoint = int(Nz/2) # mid point in discrete space

## initialise solution matrix
# each row for each different locations; each column for each time point
T = np.zeros((Nz+1, Nt+1)) 
#Tleft = 100*(np.sin(2*np.pi*t_range/period)+1)/2
#T[0,:] = Tleft

## iterate through each time point and calculate 
for i in range(1, Nt):
    # second derivative of temperature in space
    depth_2D = (T[0:-2, i-1]-2*T[1:-1,i-1]+T[2:,i-1])/dz**2
    # first derivative of temperature in time
    time_1D = K*depth_2D
    # calculate next time point temperature for each non-boundary point
    T[1:-1,i] = T[1:-1,i-1]+dt*time_1D
    T[midpoint,i] = 100*(np.sin(2*np.pi*dt*i/period)+1)/2
    # apply boundary condition on left/right end: no heat going in/out
    T[Nz,i] = T[Nz-1,i]
    T[0,i] = T[1,i]

## display the result
# we will plot heat distribution from four different time points
sampleT= int(Nt/4)
sampleTarray = sampleT*np.array([1,2,3,4])*timeTotal/Nt
sampleTarray = np.round(sampleTarray,2)
sampleTarray = map(str,sampleTarray)


# plot location vs temperature at four different time points
plt.subplot(211)
plt.plot(Z_range, T[:, 1*sampleT] , color = 'r', label= 't = '+sampleTarray[0])
plt.plot(Z_range, T[:, 2*sampleT] , color = 'g', label= 't = '+sampleTarray[1])
plt.plot(Z_range, T[:, 3*sampleT] , color = 'b', label= 't = '+sampleTarray[2])
plt.plot(Z_range, T[:, 4*sampleT] , color = 'k', label= 't = '+sampleTarray[3])
plt.ylabel('temperature')
plt.xticks([]) 
plt.legend(loc=1)

# make heatmap of temperature in location and time
plt.subplot(212)
Tshow = np.flipud(T[:, 0:].transpose())
plt.imshow(Tshow, cmap=plt.cm.hot, interpolation='nearest', origin='lower')
plt.axis('auto')
plt.axis([0, Nz+1, 0, Nt-sampleT+1])
zStep = int(Nz/5) # plot five locations on x-axis
xbound = np.arange(0, Nz+1, zStep)
xlabel = map(str, xbound*depth/Nz)
ybound = [0, Nt-sampleT]
timeStartDisplay =  round(timeTotal*sampleT/Nt, 2)
timeStartDisplay =  0
ylabel = map(str, [timeTotal, timeStartDisplay])
plt.xticks(xbound, xlabel) 
plt.yticks(ybound, ylabel)
plt.xlabel('location')
plt.ylabel('time')
# setup color bar
Tmin = Tshow.min()
Tmax = Tshow.max()
Tstep = (Tmax-Tmin)/4
colorRange = np.arange(Tmin, Tmax+Tstep, Tstep)
colorRange = np.round(colorRange, 2)
plt.colorbar(orientation='horizontal', ticks = colorRange)

plt.show()


