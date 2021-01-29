# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 02:30:12 2020

@author: ESSOUSY YOUSSEF
"""

""" 
    In this file, we call some functions from the file "Schemes.py", so make 
    sure the file "Schemes.py" exists in the same path where this file exists.
    
    The example we treat here using RBF method with Euler and Rung-Kutta's schemes
    for time stepping, is the 3D advection-diffusion equation given in the project 
    of Radial basis functions.
"""

import numpy as np
from matplotlib import pyplot as plt
import Schemes2D
import Schemes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.ticker as mtick

###################  Schémas de discrétisation temporelle  ###################

schema  =  'Euler'                    # Schéma d'Euler
schema  =  'Rung-Kutta 2'             # Schéma de Rang-Kutta 2
schema  =  'Rung-Kutta 3'             # Schéma de Rang-Kutta 3
schema  =  'Rung-Kutta 4'             # Schéma de Rang-Kutta 4

######################  Choix de la fct RBF (MQ ou CS)  ######################

method  =  'CS-RBF'                   # Approximation par la fct MQ
method  =  'MQ-RBF'                   # Approximation par la fct CS

############################  Données initiales  #############################

a, b  =  0, 1  
e, f  =  0, 1                        
g, h  =  0, 1                         # [a, b]*[e, f]*[g, h] est le domaine spatiale
t, T  =  0, 0.3                       # Le temps initial et le temps final
dt    =  0.005
Nx    =  15                          # Le nombre des points pris dans [a, b]
Ny    =  15                           # Le nombre des points pris dans [a, b]
Nz    =  15
N     =  Nx*Ny*Nz                     # Le nombre des points de collocation
c     =  4.5                          # The shape parameter pour MQ
sigma =  1                            # The shape parameter pour CS
cx    =  0.8                          # La vitesse suivant x
cy    =  0.8                          # La vitesse suivant y
cz    =  0.8
D     =  0.01                         # Le coefficient de diffusion
x     =  np.linspace(a, b, Nx)        # La subdivision uniforme de [a, b]
y     =  np.linspace(e, f, Ny)        # La subdivision uniforme de [e, f]
z     =  np.linspace(g, h, Nz)
dx    =  (b-a)/(Nx-1)                 # Le pas de discétisation dans [a, b]
dy    =  (f-e)/(Ny-1)                 # Le pas de discétisation dans [e, f]
dz    =  (h-g)/(Nz-1)                 # Le pas de discétisation dans [g, h]
cfl   =  0.5                          # La condition CFL de convergence pour le schéma "Rung-Kutta 4"
u     =  np.zeros((N, 1))             # Le vecteur Solution Approchée         
#U     =  np.zeros((Ny, Nx, Nz))      # La matrice Solution Approchée
#Uexa  =  np.zeros((Ny, Nx, Nz))      # La matrice Solution Exacte
uexa  =  np.zeros((N, 1))             # Le vecteur Solution Exacte

###################  choix de shape parameter convenable  ####################

if method == 'MQ-RBF' :
    shape = c
elif method == 'CS-RBF' :
    shape = sigma


X, Y, Z = np.meshgrid(x, y, z)


X1    =   X.reshape((np.prod(X.shape),))
Y1    =   Y.reshape((np.prod(Y.shape),))
Z1    =   Z.reshape((np.prod(Z.shape),))
 
F  =  np.zeros((N, 4))
for i in range(0, N) :                        
    F[i,0] = X1[i]  
    F[i,1] = Y1[i]
    F[i,2] = Z1[i] 


r   = np.zeros((N,N))
s   = np.zeros((N,N))
w   = np.zeros((N,N))
v   = np.zeros((N,N))                                   
A   = np.zeros((N,N))  
Ax  = np.zeros((N,N))  
Ay  = np.zeros((N,N))  
Axy = np.zeros((N,N))  


                      
l1  = [a, b] 
l2  = [e, f]
l3  = [g, h]
ctr = 0
for i in range(0, N) :
    if F[i,0] in l1 or F[i,1] in l2 or F[i,2] in l3 :  
        F[i,3] = 1
        ctr = ctr + 1
        
###################  Affichage des points de collocation  ####################
        
# fig = plt.figure(1)
# ax = fig.add_subplot(111,projection='3d')
# line = ax.scatter(X, Y, Z, s = 10, c = 'red', marker = 'o', linewidth = 0.20)
# ax.set_title("Les points de collocation pour " r"$N_x\times N_y\times N_z = 15\times 15\times 15$")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

       
    
for i in range(0, N) :
    for j in range(0, N) :
        r[i, j] = np.sqrt(sum((F[i, 0:3] - F[j, 0:3])**2))
        s[i, j] = F[i, 0] - F[j, 0]
        w[i, j] = F[i, 1] - F[j, 1]
        v[i, j] = F[i, 2] - F[j, 2]
        

############################  Condition initiale  ############################

def fct(x, y, z, t):
    k1 = -(x - 0.8*t - 0.5)**2/(0.01*(4*t + 1))
    k2 = -(y - 0.8*t - 0.5)**2/(0.01*(4*t + 1))
    k3 = -(z - 0.8*t - 0.5)**2/(0.01*(4*t + 1))
    return (1/(4*t + 1)**(3/2))*np.exp(k1 + k2 + k3)

Uexa  =  fct(X, Y, Z, t)   # Solution exacte à l'instante t = 0
U     =  Uexa                        # Les deux solutions sont égaux à l'instante t = 0
u     =  U.reshape(N, 1)


B, Bx, Bxx =  Schemes2D.choose_method_3D(method)(r, s, w, v, shape)
By, Byy    =  Schemes2D.choose_method_3D(method)(r, w, s, v, shape)[1:3]
Bz, Bzz    =  Schemes2D.choose_method_3D(method)(r, v, s, w, shape)[1:3]
Bxyz   =  Bxx + Byy + Bzz
Bxy    =  Bx + By

#Ax    =   np.dot(Bx, np.linalg.inv(B)) 
#Ay    =   np.dot(By, np.linalg.inv(B))
Axy   =   np.dot(Bxy, np.linalg.inv(B)) 
Az    =   np.dot(Bz, np.linalg.inv(B))
Axyz  =   np.dot(Bxyz, np.linalg.inv(B))


###################  Affichage des solutions initiales  ######################

                      
# fig1 = plt.figure(1)
# ax1 = Axes3D(fig1)
# surf1 = ax1.plot_surface(X, Y, Uexa, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
# fig1.colorbar(surf1)
# plt.title('Solution approchée U en t = '+str(t))
# plt.xlabel('x')
# plt.ylabel('t')
# plt.show()


############  Calcule & affichage des solutions pour 0 < t <= T  #############

Nt  = int(T/dt) + 1

error_MAE  =  []
error_RMSE =  []
list = []
Err1 = 0


while t < T :
    
    #dt_conv  =  cfl*dx/cr                   # Condition cfl pour le terme de transport
    #dt_diff  =  cfl*dx**2/(2*D)             # Condition cfl pour le terme de diffusion
    #dt       =  min(dt_conv, dt_diff)       # dt vérifie tous les deux conditions cfl
    dt       =  min(dt, T-t)                 # Pour calculer la dernière itération
    t        =  t + dt                       # Incrémentation du temps
    #t        =  round(t, 4)
    list.append(t)
    
    ###########################  Solution Exacte  ############################
    
    Uexa  =  fct(X, Y, Z, t)
    
    uexa  =  Uexa.reshape(N, 1) 
            
    #########################  Solution Approchée  ###########################
                
    f = Schemes2D.ConvDiff2D           # f sous forme de l'équation d'advection-diffusion 2D
    
    u = Schemes2D.choose_schemes(schema)(dt, cx, cy, Axy, Az, Axyz, u, D, f)
    
    #########################  Conditions limites  ###########################
    
    for i in range(0, N) :
        if F[i, 3] == 1 :
            u[i]  =  uexa[i]
    
    #####################  Solution Matrice Approchée  #######################
            
    U = u.reshape(Ny, Nx, Nz)
       
    error_MAE.append(Schemes.MAE_error(uexa, u))      # Erreurs MAE pour la méthode MQ       
    error_RMSE.append(Schemes.RMSE_error(uexa, u))
    
    Err1 = Err1 + Schemes.RMSE_error(uexa, u)**2  
    
    #####################  Affichage des deux solutions  #####################
    
    # Attention !!! These lines from 208 to 226 are not working because the display of this graph 
    # needs 4 dimensions since we have thre axes x, y and z.
    """ 
    fig = plt.figure(figsize = plt.figaspect(0.4))
    plt.subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, U, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
    plt.colorbar(surf, shrink = 0.8)
    ax1.set_title('Solution approchée U en t = '+str(t))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('U')
    
    plt.subplot(1, 2, 2)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Uexa, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
    plt.colorbar(surf, shrink = 0.8)
    ax2.set_title('Solution exacte' r' $U_{exa}$ en t = '+str(t))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('U')
    plt.show()
    """
    # Attention


error_MAE_total  = max(error_MAE)
error_RMSE_total = np.sqrt(Err1/len(list))
print("error_MAE_total = ", "{:.3e}".format(error_MAE_total))
print("error_RMSE_total = ", "{:.3e}".format(error_RMSE_total))    
    

plt.close(plt.figure(2))  
fig2  =  plt.figure(2)
ax    =  fig2.add_subplot(111)
plt.plot(list, error_MAE, 'r')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"L'évolution de l'erreur $MAE$ dans le temps")
plt.xlabel('t')
plt.ylabel(r'$MAE_{MQ}}$ et $MAE_{CS}$')
plt.legend([r"$MAE_{MQ}$", r"$MAE_{CS}$"])
plt.show()

plt.close(plt.figure(3))  
fig3  =  plt.figure(3)
ax    =  fig3.add_subplot(111)
plt.plot(list, error_RMSE, 'r')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"L'évolution de l'erreur $RMSE$ dans le temps")
plt.xlabel('t')
plt.ylabel(r'$RMSE_{MQ}}$ et $RMSE_{CS}$')
plt.legend([r"$RMSE_{MQ}$", r"$RMSE_{CS}$"])
plt.show()
