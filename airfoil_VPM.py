import os
import math
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt

V_inf = 1.0
AoA = 0
#AoA in radians
AoAR = AoA * (np.pi/180)

airfoil_filepath = os.path.join('./airfoils/naca0012.dat')
#.dat file must have its name on the first line. skiprows = [0] properly skips it
data = pd.read_table(airfoil_filepath,delim_whitespace=True, skiprows=[0], names=['x','y'],index_col=False)

def define_panels(x, y, N):
    R = (x.max() - x.min()) / 2
    
    x_center = (x.max() + x.min()) / 2
    x_circle = x_center + R*np.cos(np.linspace(0.0, 2*math.pi, N + 1))

    
    x_ends = np.copy(x_circle)

    y_ends = np.empty_like(x_ends)
    I = 0

    for i in range(N):
        while I < len(x) - 1:
            
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1


        #calculating slope of the two consecutive points        
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        #calculates the intercept 'b', from y = ax + b. it could have been used either of the two points, in this case it is used x[I+1]
        b = y[I + 1] - a * x[I + 1]
        #since we have the slope and intercept, we have the first degree equation of the two consecutive airfoil points. Since we already have the x coordinate, we can calculate the y coordinate
        y_ends[i] = a * x_ends[i] + b
        
    y_ends[N] = y_ends[0]

    panels = np.empty(N, dtype = object)
    for i in range(N):
        panels[i] = (x_ends[i], y_ends[i])

    return x_ends, y_ends

#print(define_panels(data.x, data.y, 60))

XB, YB = define_panels(data.x, data.y, 150)

#XB, YB = (data.x, data.y)

numPts = len(XB)
numPan = numPts - 1

# %%Checking panel directions

edge = np.zeros(numPan)

#if the sign of edge[i] is negative it means that the vectors are CW, otherwise they're CCW
for i in range(numPan):
    edge[i] = (XB[i + 1] - XB[i]) * (YB[i + 1] + YB[i])

sumEdge = np.sum(edge)

#Therefore, if the sum of edge[i] is less than zero, it implies that the cylinder is oriented clockwise, to which then we flip the arrays to change the orientation to CCW
if sumEdge < 0:

    XB = np.flipud(XB)

    YB = np.flipud(YB)

# %% PANEL METHOD GEOMETRY

#We now calculate the coordinates for the control points.
#Initializing arrays

XC = np.zeros(numPan)

YC = np.zeros(numPan)

S = np.zeros(numPan)

phi = np.zeros(numPan)

for i in range(numPan):
    XC[i] = 0.5 * (XB[i] + XB[i + 1])
    YC[i] = 0.5 * (YB[i] + YB[i + 1])

    dx = XB[i + 1] - XB[i]
    dy = YB[i + 1] - YB[i]

    #panel length, pythagoras
    S[i] = (dx**2 + dy**2) **0.5

    #From Anderson, ex 3.19, phi are the angles measured in the CCW direction from the x axis to the bottom of each panel
    phi[i] = math.atan2(dy, dx)
    
    #make all panel angles positive[rad]
    if phi[i] < 0:
        phi[i] = phi[i] + 2*np.pi

#angle of panel normal with respect to horizontal, and include AoA
delta = phi + (np.pi / 2)

beta = delta - AoAR

#ensures that any beta value greater than 2pi radians is reduced to its base value.
beta[beta > 2 * np.pi] = beta[beta > 2*np.pi] - 2 * np.pi


#%% COMPUTE SOURCE PANEL STRENGTHS

K = np.zeros([numPan, numPan])
L = np.zeros([numPan, numPan])

#that is, we maintain a panel fixed while we iterate through the other panels, to calculate their influence on the fixed panel

#From Anderson section 3.17, when j = i, at the control point itself r_ij = 0. Therefore, when j = i, the contribution to derivative is simply source strength/2

def COMPUTE_K_L(XC, YC, XB, YB, phi, S):
    for i in range(numPan):
        for j in range(numPan):
            if j != i:
                A = -(XC[i] - XB[j]) * np.cos(phi[j]) - (YC[i] - YB[j]) * np.sin(phi[j])
                B = (XC[i] - XB[j])**2 + (YC[i] - YB[j])**2
                C = np.sin(phi[i] - phi[j])
                D = (YC[i] - YB[j]) * np.cos(phi[i]) - (XC[i] - XB[j]) * np.sin(phi[i])
                E = np.sqrt(B - A**2)

                if (E == 0 or np.iscomplex(E) or np.isnan(E)):
                    K[i, j] = 0
                    L[i, j] = 0
                else:
                    
                    term1 = 0.5 * C * np.log((S[j]**2 + 2 * A * S[j] + B) / (B))
                    term2 = ((D - A * C) / (E)) * (math.atan2((S[j] + A), E) - math.atan2(A, E))

                    K[i, j] = term1 + term2

                    

                    term1 = ((D - A * C) / (2 * E)) * np.log((S[j]**2 + 2 * A * S[j] + B) / B )
                    term2 = C * (math.atan2((S[j] + A), E) - math.atan2(A, E)) 


                    L[i, j] = term1 - term2
            
            if (np.iscomplex(K[i, j]) or np.isnan(K[i, j]) or np.isinf(K[i, j])):
                K[i, j] = 0
            if (np.iscomplex(L[i, j]) or np.isnan(L[i, j]) or np.isinf(L[i, j])):
                L[i, j] = 0
    return K, L

K, L = COMPUTE_K_L(XC, YC, XB, YB, phi, S)

#Compute geometric integrals
#A matrix is used to solve the system of equations. 
A = np.zeros([numPan, numPan])

for i in range(numPan):
    for j in range(numPan):
        if i == j:
            #that is, the main diagonal is populated with 0
            A[i, j] = 0
        else:
            A[i, j] = -K[i, j]

#populating B array
b = np.zeros(numPan)

for i in range(numPan):
    b[i] = -V_inf * 2 * np.pi * np.cos(beta[i])

# Satisfy the Kutta Condition

pct = 100
panRep = int((pct/100)*numPan) - 1
if (panRep >= numPan):
    panRep = numPan - 1
A[panRep, :] = 0
A[panRep, 0] = 1
A[panRep, numPan - 1] = 1
b[panRep] = 0

gamma = np.linalg.solve(A, b)
print("Sum of L: ",sum(gamma*S))   

# %%COMPUTE PANEL VELOCITIES AND PRESSURE COEFFICIENTS

Vt = np.zeros(numPan)

Cp = np.zeros(numPan)

#From equation 3.155, we have the summation
for i in range(numPan):
    addVal = 0
    for j in range(numPan):
        #The tangential velocity on a plat panel induced by the panel itself is zero, hence, the term for j = i is zero. But that is already considered in the function COMPUTE_i_J
        addVal = addVal - (gamma[j] / (2*np.pi)) * L[i, j]
    
    #from eq. 3.156, we can get the total surface velocity at the ith control point
    Vt[i] = V_inf*np.sin(beta[i]) + addVal + gamma[i]/2

    #And from eq. 3.38, we get the pressure coefficient at the ith control point

    Cp[i] = 1 - (Vt[i] / V_inf)**2

# %%COMPUTE LIFT AND DRAG

#Normal force coefficient
CN = -Cp * S * np.sin(beta)
#Axial Force coefficient
CA = -Cp * S * np.cos(beta)

#Lift coefficient
CL = sum(CN * np.cos(AoAR)) - sum(CA * np.sin(AoAR))
#Drag Coefficient
CD = sum(CN * np.sin(AoAR)) + sum(CA * np.cos(AoAR))
CM = sum(Cp*(XC - 0.25)*S*np.cos(phi))

print("CL      : ",CL)                                                         
print("CD      : ",CD)                                                         
print("CM      : ",CM)   


# %% PLOTTING

def plot_airfoil(x, y):
    plt.figure()
    plt.plot(x, y, 'r',marker='.',markeredgecolor='black', markersize=3)
    plt.axis('equal')
    plt.xlim((-0.05, 1.05))
    plt.show()

plot_airfoil(data.x, data.y)

def plot_pressure():
    plt.figure()
    plt.cla()
    midIndS = int(np.floor(len(Cp)/2))
    plt.plot(XC[midIndS+1:len(XC)], Cp[midIndS+1:len(XC)], 'ks', markerfacecolor='b', label='VPM Upper')
    plt.plot(XC[0:midIndS], Cp[0:midIndS], 'ks', markerfacecolor='r', label='VPM Lower')
    plt.xlim(0,1)
    #plt.ylim(-2, 2)
    plt.xlabel('X Coordinate')
    plt.ylabel('Cp')
    plt.title('Pressure Coefficient')
    plt.show()
    plt.legend()
    plt.gca().invert_yaxis()

plot_pressure()
