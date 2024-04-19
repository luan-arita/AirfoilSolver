import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns


V_inf = 1.0
AoA = 0
#AoA in radians
AoAR = AoA * (np.pi/180)
numB = 8 #number of boundary points, that is, number of panel extremities
R = 1 #radius

#gets the panel points by dividing a circle by the number of boundary points
theta = np.linspace(0, 360, numB)
#converts theta to radians
theta = theta*(np.pi / 180)

#X-coordinate for the panel
XB = np.cos(theta)
#Y-coordinate for the panel
YB = np.sin(theta)

numPan = len(XB) - 1

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

#generates 2d arrays for I and J, which will represent the panels and its respective source strength
I = np.zeros([numPan, numPan])
J = np.zeros([numPan, numPan])

#that is, we maintain a panel fixed while we iterate through the other panels, to calculate their influence on the fixed panel

#From Anderson section 3.17, when j = i, at the control point itself r_ij = 0. Therefore, when j = i, the contribution to derivative is simply source strength/2

def COMPUTE_I_J(XC, YC, XB, YB, phi, S):
    for i in range(numPan):
        for j in range(numPan):
            if j != i:
                A = -(XC[i] - XB[j]) * np.cos(phi[j]) - (YC[i] - YB[j]) * np.sin(phi[j])
                B = (XC[i] - XB[j])**2 + (YC[i] - YB[j])**2
                C = np.sin(phi[i] - phi[j])
                D = (YC[i] - YB[j]) * np.cos(phi[i]) - (XC[i] - XB[j]) * np.sin(phi[i])
                E = np.sqrt(B - A**2)

                if (E == 0 or np.iscomplex(E) or np.isnan(E)):
                    I[i, j] = 0
                    J[i, j] = 0
                else:
                    #calculating I using coefficients, described in eq. 3.163
                    term1 = 0.5 * C * np.log((S[j]**2 + 2 * A * S[j] + B) / (B))
                    term2 = ((D - A * C) / (E)) * (math.atan2((S[j] + A), E) - math.atan2(A, E))

                    I[i, j] = term1 + term2

                    #Now calculating J, needed for tangential velocity and described in eq. 3.165

                    term1 = ((D - A * C) / (2 * E)) * np.log((S[j]**2 + 2 * A * S[j] + B) / B )
                    term2 = C * (math.atan2((S[j] + A), E) - math.atan2(A, E)) 


                    J[i, j] = term1 - term2
            
            if (np.iscomplex(I[i, j]) or np.isnan(I[i, j]) or np.isinf(I[i, j])):
                I[i, j] = 0
            if (np.iscomplex(J[i, j]) or np.isnan(J[i, j]) or np.isinf(J[i, j])):
                J[i, j] = 0
    return I, J

I, J = COMPUTE_I_J(XC, YC, XB, YB, phi, S)

#Compute geometric integrals
#A matrix is used to solve the system of equations. 
A = np.zeros([numPan, numPan])

for i in range(numPan):
    for j in range(numPan):
        if i == j:
            #that is, the main diagonal is populated with pi
            A[i, j] = np.pi
        else:
            A[i, j] = I[i, j]

#populating B array
b = np.zeros(numPan)

for i in range(numPan):
    b[i] = -V_inf * 2 * np.pi * np.cos(beta[i])

#solving system of equations and obtaining source panel strengths(lam is from lambda)

lam = np.linalg.solve(A, b)
            

#From example 3.19, the accuracy of our results can be tested by the sum of strengths. For a closed body, the sum must be equal to zero
print("Sum of L: ",sum(lam*S))   

# %%COMPUTE PANEL VELOCITIES AND PRESSURE COEFFICIENTS

Vi = np.zeros(numPan)

Cp = np.zeros(numPan)

#From equation 3.155, we have the summation
for i in range(numPan):
    addVal = 0
    for j in range(numPan):
        #The tangential velocity on a plat panel induced by the panel itself is zero, hence, the term for j = i is zero. But that is already considered in the function COMPUTE_i_J
        addVal = addVal + (lam[j] / (2*np.pi)) * J[i, j]
    
    #from eq. 3.156, we can get the total surface velocity at the ith control point
    Vi[i] = V_inf*np.sin(beta[i]) + addVal

    #And from eq. 3.38, we get the pressure coefficient at the ith control point

    Cp[i] = 1 - (Vi[i] / V_inf)**2

#For calculating the analytic results for comparison, we get here the analytical angles and pressure coefficients
analyticTheta = np.linspace(0, 2*np.pi, 200)
analyticCP = 1 - 4*np.sin(analyticTheta)**2

# %%COMPUTE LIFT AND DRAG

#Normal force coefficient
CN = -Cp * S * np.sin(beta)
#Axial Force coefficient
CA = -Cp * S * np.cos(beta)

#Lift coefficient
CL = sum(CN * np.cos(AoAR)) - sum(CA * np.sin(AoAR))
#Drag Coefficient
CD = sum(CN * np.sin(AoAR)) + sum(CA * np.cos(AoAR))

print("CL      : ",CL)                                                          # Display lift coefficient (should be zero)
print("CD      : ",CD)                                                          # Display drag coefficient (should be zero)


#Plotting pressure coefficient and comparing with analytical results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)                                    
plt.plot(analyticTheta*(180/np.pi),analyticCP,'deepskyblue',label='Analítico')    
plt.plot(beta*(180/np.pi),Cp,'^',markerfacecolor='black',label='Source Panel Method', markersize = 10)
plt.xlabel('Ângulo [deg]')                                                   
plt.ylabel('Cp')                                          
plt.title('Distribuição de Pressão Sobre um Cilindro')                                
plt.xlim(0, 360)                                                            
plt.ylim(-3.5, 1.5)
plt.legend()                                                                
plt.show()                                                                  


def cylinder(radius):
    x_center, y_center = 0.0, 0.0

    #linspace creates evenly spaced values within [start, stop, number]
    #this is splitting the circle in 100 parts
    theta = np.linspace(0.0, 2 * math.pi, 100)
    x_cylinder = x_center + radius * np.cos(theta)
    y_cylinder = y_center +radius * np.sin(theta)


    return x_cylinder, y_cylinder

def plot_cylinder():

    x_cylinder, y_cylinder = cylinder(R)

    size = 6
    plt.figure(figsize = (size, size))
    plt.grid()
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.plot(x_cylinder, y_cylinder)

    plt.show()

class Panel:
    def __init__(self, xa, ya, xb, yb):
        
        #initial coordinates of panel
        self.xa, self.ya = xa, ya
        #final coordinates of panel
        self.xb, self.yb = xb, yb

        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2 #control point
        self.length = math.sqrt((xb - xa)**2 + (yb - ya)**2) #panel length

        #angle between x-axis and panel's normal
        if xb - xa <= 0.: #that is, the panel is located on the lower half of the cylinder
            self.beta = math.acos((yb - ya)) / self.length
        elif xb - xa > 0.: #that is, the panel is located on the upper half of the cylinder
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)

        self.sigma = 0.0 #source strength
        self.vt = 0.0 #tangential velocity
        self.cp = 0.0 #pressure coefficient



def panels_array(radius, N_panels):
    x_ends = radius * np.cos(np.linspace(0.0, 2*math.pi, N_panels + 1)) 
    y_ends = radius * np.sin(np.linspace(0.0, 2*math.pi, N_panels + 1)) 

    panels = np.empty(N_panels, dtype = object)
    for i in range(N_panels):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])

    return panels, x_ends, y_ends

def plot_panels_array(numB):
    
    x_cylinder, y_cylinder = cylinder(R)
    panels, x_ends, y_ends = panels_array(R, numB)

    size = 6
    plt.figure(figsize=(size, size))
    plt.grid()
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.plot(x_cylinder, y_cylinder,
                label='cylinder',
                color='b', linestyle='-', linewidth=1)
    plt.plot(x_ends, y_ends,
                label='panels',
                color='#CD2305', linestyle='-', linewidth=2)
    plt.scatter([p.xa for p in panels], [p.ya for p in panels],
                label='end-points',
                color='#CD2305', s=40)
    plt.scatter([p.xc for p in panels], [p.yc for p in panels],
                label='center-points',
                color='k', s=40, zorder=3)
    plt.legend(loc='best', prop={'size':12})
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1);
    plt.show()

plot_panels_array(numB)
