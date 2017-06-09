import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

start = time.time()
# Here we study the progress of 1D, two species stripe pattern


## initialise parameters of heat equation
depth = 20. # in this case the width in mm of space to generate  pattern
celSize = 1. # this is expected size of the seeding band
timeTotal = 1500.

D = .001 # diffusion rate of s1 and s2 signal
Ka = .2 # maximal switching rate of c0->c1s and c0->c2s
Km = 100 # [s2] and [s1] at half maximal swiching rate of c0->c1s and c0->c2s
A = 10000 # production rate of s1 and s2 from c1 and c2
Ga = .1 # degradation rate constant of s1 and s2
Tau = 15# the delay time unit for c1s->c1 and c2s->c2 conversion

#signal saturation level for making heatmap
satLevel = Km*2

## specify parameter for numerical solution
Nz = 800 # number of discrete step in the space of pattern forming domain 
dz = depth/Nz # size of discrete step in space
cel = int(celSize/dz) # number of discrete step in seeding band
# array of each location in our discrete space
dt = 0.9*dz**2/2/D # size of discrete step in time, note that for Explicit Euler, we need dt<=dz**2/2/D for stability of numerical solution
Nt = int(timeTotal/dt) # total number of discrete time points in simulation
NTau = int(Tau/dt) # total number of discrete time for production delay.
Z_range = np.arange(0,depth+dz,dz) # space index for the space  domain
t_range = np.arange(0, timeTotal, dt) # time index for the whole simulation

print('dt = ' + str(dt))

## initialise solution matrix
# each row for each different locations; each column for each time point
S1 = np.zeros((Nz+1, Nt+1)) # signalling molecule S1
S2 = np.zeros((Nz+1, Nt+1)) # signalling molecule S2
C0 = np.zeros((Nz+1, Nt+1)) # cell type C0 (progenitor cell)
C1 = np.zeros((Nz+1, Nt+1)) # cell type C1 (differentiated cell type-1)
C2 = np.zeros((Nz+1, Nt+1)) # cell type C2 (differentiate cell type-2)
C1s = np.zeros((Nz+1, NTau+1)) # newly formed C1 not capable of sending S1
C2s = np.zeros((Nz+1, NTau+1)) # newly formed C2 not capable of sending S2

C1[0:cel+1, :] = 1. #at simulation starts, all cell in the seeding band is C1
C0[cel+1:, 0] = 1. #at simulation starts, all other cell in the domain is C0


## iterate through each time point and compute the amount of different cell types and different signalling molecules

for i in range(1,  Nt+1):
    # second derivative of each diffusible species in space
    depthS1_2D = (S1[0:-2, i-1]-2*S1[1:-1,i-1]+S1[2:,i-1])/dz**2
    depthS2_2D = (S2[0:-2, i-1]-2*S2[1:-1,i-1]+S2[2:,i-1])/dz**2

    S1c = S1[1:-1,i-1]
    S2c = S2[1:-1,i-1]
    # first derivative of each species concentration in time
    # (excluding production & degradation)
    timeS1_1D = D*depthS1_2D
    timeS2_1D = D*depthS2_2D

    # calculate the amount of C1s cells at different maturation stages
    outC1s = np.copy(C1s[:, NTau]) # C1s cell matures & read to become  C1
    oldC1s = np.copy(C1s[:, 0:NTau])
    #C0 entering C1s state at the rate depending on S2 level
    inC1s = dt*C0[:,i-1]*S2[:,i-1]*Ka/(S2[:,i-1]+Km) 
    # update the amount of C1s cell at different maturation stages
    C1s[:,0] = np.copy(inC1s)
    C1s[:,1:NTau+1] = np.copy(oldC1s)

    # calculate the amount of C2s cells at different maturation stages
    outC2s = np.copy(C2s[:, NTau]) # C2s cell matures & read to become  C2
    oldC2s = np.copy(C2s[:, 0:NTau])
    #C0 entering C2s state at the rat depending on S1 level
    inC2s = dt*C0[:,i-1]*S1[:,i-1]*Ka/(S1[:,i-1]+Km) 
    # update the amount of C1s cell at different maturation stages 
    C2s[:,0] = np.copy(inC2s)
    C2s[:,1:NTau+1] = np.copy(oldC2s)

    # calculate the amount of C0, C1 and C2 at the next time step
    C0[:,i] =  C0[:,i-1]-inC1s-inC2s # C0 is lost going into C1s or C2s
    C1[:,i] =  C1[:,i-1]+outC1s  # C1 gained from matured C1s
    C2[:,i] =  C2[:,i-1]+outC2s # C2 gained from matured C2s

    # calculate next time point S1, S2 concentrations
    # change due to diffusion
    S1[1:-1,i] = S1c + dt*timeS1_1D 
    S2[1:-1,i] = S2c + dt*timeS2_1D
    # change due to production and degradation 
    S1[1:-1,i] = S1[1:-1,i] + dt*A*C1[1:-1,i] - dt*Ga*S1[1:-1,i]
    S2[1:-1,i] = S2[1:-1,i] + dt*A*C2[1:-1,i] - dt*Ga*S2[1:-1,i]
    
    # apply Neumann (insulated) boundary condition on both ends 
    S1[Nz,i] = S1[Nz-1,i]
    S2[Nz,i] = S2[Nz-1,i]
    S1[0,i] = S1[1,i]
    S2[0,i] = S2[1,i]

## display the result
#-------------------------------------------------------------------------
## The PLOT-1  will show the amount of each cell types at the end of simulation
# -------------------------------------------------------------------------

# get the amount of cells in C0, C1 and C2 state at the end of simulation 
C0final =  np.around(C0[:, -1], decimals = 3)
C1final = np.around(C1[:, -1], decimals = 3)
C2final =  np.around(C2[:, -1], decimals = 3)

# calculate the total amount of C1s and C2s state cells. 
# add up all cells in different maturation state of C1s or C2s, round to 3 decimal.
C1sSum = np.around(np.sum(C1s,axis=1), decimals = 3)
C2sSum = np.around(np.sum(C2s,axis=1), decimals = 3)

# Generate plots
plt.subplot(411) #fraction of different cell types at the final time point
plt.plot(C0final,color='k')
plt.plot(C1final,color='r')
plt.plot(C2final,color='b')
plt.plot(C1sSum,color='r', linestyle='--')
plt.plot(C2sSum,color='b', linestyle='--')
plt.ylabel('fraction')
plt.axis([0, Nz+1, -0.1, 1.1])
plt.xticks([])
# ------------------------------------------------------------------------
# This next three plots will show heatmaps of each signals and cell types at different
# location and time point.

# reconfigure concentration matrix of S1, S2, C1, C2 so that
# we display time on Y-axis starting from top to bottom
# and display space on X-axis from left to right
S1show = np.flipud(S1.transpose())
S2show = np.flipud(S2.transpose())
C1show = np.flipud(C1.transpose())
C2show = np.flipud(C2.transpose())

#-------------------------------------------------------------------------
## The PLOT-2  will show S1 level over time/space, more white == higher level
# -------------------------------------------------------------------------
plt.subplot(412) 
plt.imshow(S1show, cmap=plt.cm.gray, interpolation='nearest', origin='lower', vmin=0, vmax=satLevel)
plt.axis('auto')
plt.axis([0, Nz+1, 0, Nt+1])
plt.xticks([])
plt.yticks([0, Nt], [str(timeTotal), str(0)])
plt.ylabel('time')
#-------------------------------------------------------------------------
## The PLOT-3  will show S2 level over time/space, more white == higher level
# -------------------------------------------------------------------------
plt.subplot(413) 
plt.imshow(S2show, cmap=plt.cm.gray, interpolation='nearest', origin='lower', vmin=0, vmax=satLevel)
plt.axis('auto')
plt.axis([0, Nz+1, 0, Nt+1])
plt.xticks([])
plt.yticks([0, Nt], [str(timeTotal), str(0)])
plt.ylabel('time')
# -------------------------------------------------------------------------
## The PLOT-4  will show C1 and C2 level over time/space in red and blue, repectively
# -------------------------------------------------------------------------
#make image from C1 and C2 level
pp = C1show.shape#get the size of C1 matrix
#make a 3D zero matrix for an image representing C1 and C2 level
#the matrix has the same width and length as C1 (or C2) matrix and has
#depth = 3 representing RGB color
C1C2 = np.zeros((pp[0], pp[1], 3))
C1C2[:, :, 0] = np.copy(C1show) #C1 level represented in red
C1C2[:, :, 2] = np.copy(C2show) #C2 level represented in blue

# specify how to label x axis of the heatmap plots
zStep = int(Nz/5) # plot five locations on x-axis
xbound = np.arange(0, Nz+1, zStep)
xlabel = map(str, np.around(xbound*depth/Nz, decimals = 1))

# make heatmap of C1 and C2 w.r.t location and time
plt.subplot(414)
plt.imshow(C1C2, vmin =0, vmax =1)
plt.axis('auto')
plt.axis([0, Nz+1, 0, Nt+1])
plt.xticks(xbound, xlabel) 
plt.yticks([0, Nt], [str(timeTotal), str(0)])
plt.xlabel('location (mm)')
plt.ylabel('time (min)')

end = time.time()

# display total time to run and display simulation
print('runtime = ' + str(end-start))
plt.show()


