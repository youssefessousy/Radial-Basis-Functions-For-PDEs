# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:16:43 2020

@author: ESSOUSY YOUSSEF
"""

""" 
    In this file, we call some functions from the file "Schemes_SW.py", so make 
    sure the file "Schemes_SW.py" exists in the same path where this file exists.
    
    The example we treat here using RBF method with Euler and Rung-Kutta's schemes
    for time stepping, is the linear shallow water equation given in the project 
    of Radial basis functions.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.ticker as mtick
import Schemes
import Schemes2D
import Schemes_SW

######################  Choix de la fct RBF (MQ ou CS)  ######################

method  =  'CS-RBF'                   # Approximation par la fct MQ
method  =  'MQ-RBF'                   # Approximation par la fct CS

############################  Données initiales  #############################

L, l  =  872, 50                      # [a, b] est le domaine spatiale
t, T  =  0, 600                       # Le temps initial et le temps final
Nx    =  60                           # Le nombre des points pris dans [a, b]
Ny    =  10
N     =  Nx*Ny
g     =  9.81
x     =  np.linspace(0, L, Nx)        # La subdivision uniforme de [0, L]
y     =  np.linspace(0, l, Ny)        # La subdivision uniforme de [0, l]
dx    =  L/(Nx-1)                     # Le pas de discétisation spatiale suivant x
dy    =  l/(Ny-1)                     # Le pas de discétisation spatiale suivant y
cfl   =  0.8                          # La condition CFL de convergence
H     =  20
dmin  =  min(dx, dy)
c     =  0.8*np.sqrt(N)/dmin          # The shape parameter pour MQ
sigma =  120   

###################  choix de shape parameter convenable  ####################

if method == 'MQ-RBF' :
    shape = c
elif method == 'CS-RBF' :
    shape = sigma
    
########  Les solutions exactes et approchées sous forme matricielle  ######## 
    
nuex  =  np.zeros((Ny, Nx))   
NUex  =  np.zeros((N, 1))
uex   =  np.zeros((Ny, Nx))  
Uex   =  np.zeros((N, 1))
vex   =  np.zeros((Ny, Nx))  
Vex   =  np.zeros((N, 1))
nu    =  np.zeros((Ny, Nx))  
NU    =  np.zeros((N, 1))
u     =  np.zeros((Ny, Nx))  
U     =  np.zeros((N, 1))
v     =  np.zeros((Ny, Nx))  
V     =  np.zeros((N, 1))


Z     =  np.zeros((N,3))
r     =  np.zeros((N,N))
s     =  np.zeros((N,N))
w     =  np.zeros((N,N))                                   
A     =  np.zeros((N,N))  
Ax    =  np.zeros((N,N))  
Ay    =  np.zeros((N,N))  
Axy   =  np.zeros((N,N))  

 
############################  Condition initiale  ############################

X, Y  =   np.meshgrid(x, y)
X1    =   X.reshape((np.prod(X.shape),))
Y1    =   Y.reshape((np.prod(Y.shape),))
  
for i in range(0, N) :                        
    Z[i,0] = X1[i]  
    Z[i,1] = Y1[i] 
                      
list1  = [0, L] 
list2  = [0, l]
for i in range(0, N) :
    if Z[i,0] in list1 or Z[i,1] in list2:  
        Z[i,2] = 1
        
###################  Affichage des points de collocation  ####################
      
# for i in range(0, N) :
#     if Z[i,2] == 0:
#         p1 = plt.scatter(Z[i,0], Z[i, 1], label='Points interieurs', s = 10, c = 'red', linewidth = 0.20)
#     else :
#         p2 = plt.scatter(Z[i,0], Z[i, 1], label='Points du bord', s = 10, c = 'blue', linewidth = 0.20)
# plt.title("Les points de collocation pour " r"$N_x\times N_y = 60\times 10$")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

####################  Calcul des matrices des distances  #####################
            
for i in range(0, N) :
    for j in range(0, N) :
        r[i, j] = np.sqrt(sum((Z[i, 0:2] - Z[j, 0:2])**2)) # La distance Euclidienne
        s[i, j] = Z[i, 0] - Z[j, 0]                        # La matrice (x_i - x_j)
        w[i, j] = Z[i, 1] - Z[j, 1]                        # La matrice (y_i - y_j)



for i in range(0, Nx) :
    for j in range(0, Ny) :     
        [nuex[j, i], uex[j, i], vex[j, i]]  = Schemes_SW.Sol_Shallow_Water(x[i], y[j], t)  

NUex =  nuex.reshape(N, 1)  
Uex  =  uex.reshape(N, 1)  
Vex  =  vex.reshape(N, 1)  

B, Bx = Schemes2D.choose_method(method)(r, s, w, shape)[0:2]
By    = Schemes2D.choose_method(method)(r, w, s, shape)[1]


Ax    =   np.dot(Bx, np.linalg.inv(B)) 
Ay    =   np.dot(By, np.linalg.inv(B))

NU  =  NUex
U   =  Uex
V   =  Vex
     
W    = [NU, U, V] 
   
F   =  [-H*(np.dot(Ax, Uex)+np.dot(Ay, Vex)), -g*np.dot(Ax, NUex), -g*np.dot(Ay, NUex)]



# nu0 = 1
# w = 1.45444*0.12
# # l = 50                # Not needed for the analytical solution.
# L = 872
# H = 20
# g = 9.81
# def nuhh(x, y, t) :
#     nu = nu0*np.cos(w*(L - x)/np.sqrt(g*H))*np.cos(w*t)/np.cos(w*L/np.sqrt(g*H))
#     #u  = -nu0*np.sqrt(g/H)*np.sin(w*(L - x)/np.sqrt(g*H))*np.sin(w*t)/np.cos(w*L/np.sqrt(g*H))
#     #v  = 0
#     return nu

# UUU = nuhh(X, Y, t)

###################  Affichage des solutions initiales  ######################
                  
# fig1  =  plt.figure(1)
# ax1   =  Axes3D(fig1)
# surf1 =  ax1.plot_surface(X, Y, U, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig1.colorbar(surf1)
# plt.title('Solution approchée U en t = '+str(t))
# plt.xlabel('x')
# plt.ylabel('t')
# plt.show()

###########################  Erreurs MAE et RMSE  ##############################

nu_MAE  =  []
nu_RMSE =  []
u_MAE   =  []
u_RMSE  =  []
v_MAE   =  []
v_RMSE  =  []

error_MAE = []
error_RMSE = []
Err1  = 0
Err2  = 0
Err3  = 0

list = []
dt  =  float(cfl*dmin/max(np.linalg.norm(U) + np.sqrt(g*H), np.linalg.norm(V) + np.sqrt(g*H)))
nu0 = 1
while t < T :
    
    t    =  t + dt
    t    =  round(t, 2)
    list.append(t)
    
    ###########################  Solution Exacte  ############################
    
    for i in range(0, Nx) :
        for j in range(0, Ny) :     
            nuex[j, i], uex[j, i], vex[j, i] = Schemes_SW.Sol_Shallow_Water(x[i], y[j], t)
            #nuex[j, i] =  nu0*np.cos(w*(L - x)/np.sqrt(g*H))*np.cos(w*t)/np.cos(w*L/np.sqrt(g*H))
    
    NUex    =  nuex.reshape(N, 1)
    Uex     =  uex.reshape(N, 1)
    Vex     =  vex.reshape(N, 1)
    
    #W   =  (NUex, Uex, Vex)
 
    #########################  Solution Approchée  ###########################
    
    f  =  Schemes_SW.F
    
    NU, U, V  =  Schemes_SW.Time_Scheme(NU, U, V, Ax, Ay, dt, f)
    
    for i in range(0, N) :
        if Z[i, 2] == 1 :
            NU[i]  =  NUex[i]
            U[i]   =  Uex[i]
            V[i]   =  Vex[i]
            
            
    nu = NU.reshape(Ny, Nx)
    u  = U.reshape(Ny, Nx)  
    v  = V.reshape(Ny, Nx)  
     
    #########################  Calcul des erreurs  ###########################
    
    nu_MAE.append(Schemes.MAE_error(NU, NUex))      # Erreurs MAE pour la méthode MQ       
    nu_RMSE.append(Schemes.RMSE_error(NU, NUex))
    u_MAE.append(Schemes.MAE_error(U, Uex))
    u_RMSE.append(Schemes.RMSE_error(U, Uex))
    v_MAE.append(Schemes.MAE_error(V, Vex))
    v_RMSE.append(Schemes.RMSE_error(V, Vex))
     
    
    Err1 = Err1 + Schemes.RMSE_error(NU, NUex)**2
    Err2 = Err2 + Schemes.RMSE_error(U, Uex)**2
    Err3 = Err3 + Schemes.RMSE_error(V, Vex)**2
    
    #######################  Affichage des solutions  ########################
    
    fig = plt.figure(figsize = plt.figaspect(0.5))
    plt.subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, u, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
    plt.colorbar(surf, shrink = 0.8)
    ax1.set_title(r'$U_{CS}$ en t = '+str(t))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('U')
    
    plt.subplot(1, 2, 2)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, uex, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
    plt.colorbar(surf, shrink = 0.8)
    ax2.set_title(r'$U_{ex}$ en t = '+str(t))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('r$U$')
    plt.show()
    
    #################  Affichage des solutions en t=300  #####################
    
    # if t == 300.16 :
    #     fig1  =  plt.figure(1)
    #     ax    =  fig1.add_subplot(111)
    #     plt.plot(x, u[4, :], 'r-o', x, uex[4, :], 'b')
    #     #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #     plt.title(r"Section y = 22.22")
    #     plt.xlabel('x')
    #     plt.ylabel(r"$U_{ex}\;\text{et}$U_{ap}$")
    #     plt.show()
        
###############################  Errors Values  ##############################    
        
NU_MAE_total  =  max(nu_MAE)
NU_RMSE_total =  np.sqrt(Err1/(len(list)))
U_MAE_total   =  max(u_MAE)
U_RMSE_total  =  np.sqrt(Err2/(len(list)))
V_MAE_total   =  max(v_MAE)
V_RMSE_total  =  np.sqrt(Err3/(len(list)))

print("NU_MAE_total = ", "{:.3e}".format(NU_MAE_total))
print("NU_RMSE_total = ", "{:.3e}".format(NU_RMSE_total))  
print("U_MAE_total = ", "{:.3e}".format(U_MAE_total))
print("U_RMSE_total = ", "{:.3e}".format(U_RMSE_total))  
print("V_MAE_total = ", "{:.3e}".format(V_MAE_total))
print("V_RMSE_total = ", "{:.3e}".format(V_RMSE_total))  
  
################################  Errors Plot  ###############################

plt.close(plt.figure(2))  
fig2  =  plt.figure(2)
ax    =  fig2.add_subplot(111)
plt.plot(list, nu_MAE, 'r--')
plt.plot(list, u_MAE, 'c--')
plt.plot(list, v_MAE, 'b--')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"Les erreurs $MAE$")
plt.xlabel('t')
plt.legend([r"$MAE_{MQ}$", r"$MAE_{CS}$"])
plt.show()

plt.close(plt.figure(3))  
fig3  =  plt.figure(3)
ax    =  fig3.add_subplot(111)
plt.plot(list, nu_RMSE, 'r')
plt.plot(list, u_RMSE, 'c')
plt.plot(list, v_RMSE, 'b')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"Les erreurs $RMSE$")
plt.xlabel('t')
plt.legend([r"$\eta$", r"$u$"])
plt.show()

