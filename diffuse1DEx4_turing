import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

## This script shows how to solve 1D heat equation numerically.
# Here we study the progress of 1D 2 species (U, V) turing pattern
# we modify this from 2D turing example presented in
# http://rsfs.royalsocietypublishing.org/content/2/4/487

## initialise parameters of heat equation
depth = 10. # in this case the width of space to generate turing pattern
timeTotal = 20.


k_u = .05 # diffusion rate of u
k_v = 2 # diffusion rate of v
c1 = 0.1
c2 = 0.9
cm1 = 1
c3 = 1



## specify parameter for numerical solution
Nz = 100. # number of discrete step in space
dz = depth/Nz # size of discrete step in space
# array of each location in our discrete space
dt = 0.9*dz**2/2/max(k_u,k_v) # size of discrete step in time, note that dt<=dz**2/4 for stability of numerical solution
Z_range = np.arange(0,depth+dz,dz)
t_range = np.arange(0, timeTotal, dt)
Nt = int(timeTotal/dt) # total number of discrete time points
midpoint = int(Nz/2) # mid point in discrete space

## initialise solution matrix
# each row for each different locations; each column for each time point
U = np.zeros((Nz+1, Nt+1))
V = np.zeros((Nz+1, Nt+1))
U[:,0] = np.random.rand(Nz+1) # random  initial concentration of U
V[:,0] = np.random.rand(Nz+1) # randome initial concentration of V

## iterate through each time point and calculate 
for i in range(1, Nt):
    # second derivative of each species in space
    depthU_2D = (U[0:-2, i-1]-2*U[1:-1,i-1]+U[2:,i-1])/dz**2
    depthV_2D = (V[0:-2, i-1]-2*V[1:-1,i-1]+V[2:,i-1])/dz**2

    Uc = U[1:-1,i-1]
    Vc = V[1:-1,i-1]
    # first derivative of each species concentration in time
    timeU_1D = k_u*depthU_2D + c1 - cm1*Uc + c3*Vc*Uc**2
    timeV_1D = k_v*depthV_2D + c2 - c3*Vc*Uc**2
    
    # calculate next time point concentrations for each non-boundary point
    U[1:-1,i] = Uc + dt*timeU_1D
    V[1:-1,i] = Vc + dt*timeV_1D
    # apply boundary condition on both ends saying that nothing going in/out
    U[Nz,i] = U[Nz-1,i]
    V[Nz,i] = V[Nz-1,i]
    U[0,i] = U[1,i]
    V[0,i] = V[1,i]

## display the result
# we will plot heat distribution from four different time points
sampleT= int(Nt/4)
sampleTs = sampleT*np.array([0.5, 2, 3 ,4])
sampleTarray = sampleTs*timeTotal/Nt
sampleTarray = np.round(sampleTarray,2)
strTarray = map(str,sampleTarray)



# plot location vs U at four different time points
plt.subplot(211)
plt.plot(Z_range, U[:,sampleTs[0]],color='r',label= 't = '+strTarray[0])
plt.plot(Z_range, U[:,sampleTs[1]],color='g',label= 't = '+strTarray[1])
plt.plot(Z_range, U[:,sampleTs[2]],color='b',label= 't = '+strTarray[2])
plt.plot(Z_range, U[:,sampleTs[3]],color= 'k',label= 't = '+strTarray[3])
plt.ylabel('concentration')
plt.xticks([]) 
plt.legend(loc=1)



# make heatmap of U in location and time
plt.subplot(212)
Ushow = np.flipud(U[:, 0:].transpose())
plt.imshow(Ushow, cmap=plt.cm.hot, interpolation='nearest', origin='lower')
plt.axis('auto')
plt.axis([0, Nz+1, 0, Nt-sampleT+1])
zStep = int(Nz/5) # plot five locations on x-axis
xbound = np.arange(0, Nz+1, zStep)
xlabel = map(str, xbound*depth/Nz)
ybound = [0, Nt]
timeStartDisplay =  round(timeTotal*sampleT/Nt, 2)
timeStartDisplay =  0
ylabel = map(str, [timeTotal, timeStartDisplay])
plt.xticks(xbound, xlabel) 
plt.yticks(ybound, ylabel)
plt.xlabel('location')
plt.ylabel('time')
# setup color bar
Umin = Ushow.min()
Umax = Ushow.max()
Ustep = (Umax-Umin)/4
colorRange = np.arange(Umin, Umax+Ustep, Ustep)
colorRange = np.round(colorRange, 2)
plt.colorbar(orientation='horizontal', ticks = colorRange)

plt.show()

#print U[:, 0]

#print U[:, 2]

#print U[:, Nt]
