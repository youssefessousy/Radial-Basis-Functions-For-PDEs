"""
Created on Mon Aug 17 02:15:30 2020

@author: ESSOUSY YOUSSEF
"""

##############################################################################

""" 
    In this file, we call some functions from the file "Schemes.py", so make 
    sure the file "Schemes.py" exists in the same path where this file exists.
    
    The example we treat here using RBF method with Euler and Rung-Kutta's schemes
    for time stepping, is the 2D advection-diffusion equation given in the project 
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

a, b  =  -1, 1  
e, f  =  -1, 1                        # [a, b]*[e, f] est le domaine spatiale
t, T  =  0, 1                         # Le temps initial et le temps final
dt    =  0.01
Nx    =  30                           # Le nombre des points pris dans [a, b]
Ny    =  30                           # Le nombre des points pris dans [a, b]
N     =  Nx*Ny                        # Le nombre des points de collocation
c     =  3                            # The shape parameter pour MQ
sigma =  1                            # The shape parameter pour CS
cx    =  1                            # La vitesse suivant x
cy    =  1                            # La vitesse suivant y
D     =  0.01                         # Le coefficient de diffusion
x     =  np.linspace(a, b, Nx)        # La subdivision uniforme de [a, b]
y     =  np.linspace(e,f, Ny)         # La subdivision uniforme de [e, f]
dx    =  (b-a)/(Nx-1)                 # Le pas de discétisation dans [a, b]
dy    =  (f-e)/(Ny-1)                 # Le pas de discétisation dans [e, f]
cfl   =  0.5                          # La condition CFL de convergence pour le schéma "Rung-Kutta 4"
u     =  np.zeros((N, 1))             # Le vecteur Solution Approchée         
U     =  np.zeros((Ny, Nx))           # La matrice Solution Approchée
Uexa  =  np.zeros((Ny, Nx))           # La matrice Solution Exacte
uexa  =  np.zeros((N, 1))             # Le vecteur Solution Exacte


Z   = np.zeros((N,3))
r   = np.zeros((N,N))
s   = np.zeros((N,N))
w   = np.zeros((N,N))                                   
A   = np.zeros((N,N))  
Ax  = np.zeros((N,N))
Ay  = np.zeros((N,N))  
Axy = np.zeros((N,N))  

###################  choix de shape parameter convenable  ####################

if method == 'MQ-RBF' :
    shape = c
elif method == 'CS-RBF' :
    shape = sigma
    
############################################################################## 
    
X, Y  =   np.meshgrid(x, y)
X1    =   X.reshape((np.prod(X.shape),))
Y1    =   Y.reshape((np.prod(Y.shape),))
  
for i in range(0, N) :                        
    Z[i,0] = X1[i]  
    Z[i,1] = Y1[i] 
                      
list1  = [a, b] 
list2  = [e, f]
for i in range(0, N) :
    if Z[i,0] in list1 or Z[i,1] in list2:  
        Z[i,2] = 1

###################  Affichage des points de collocation  ####################
        
# for i in range(0, N) :
#     if Z[i,2] == 0:
#         p1 = plt.scatter(Z[i,0], Z[i, 1], label='Points interieurs', s = 15, c = 'red', linewidth = 0.20)
#     else :
#         p2 = plt.scatter(Z[i,0], Z[i, 1], label='Points du bord', s = 15, c = 'blue', linewidth = 0.20)
# plt.title("Les points de collocation pour " r"$N_x\times N_y = 30\times 30$")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

###################  Calcule des matrices des distances  #####################
   
for i in range(0, N) :
    for j in range(0, N) :
        r[i, j] = np.sqrt(sum((Z[i, 0:2] - Z[j, 0:2])**2))
        s[i, j] = Z[i, 0] - Z[j, 0]
        w[i, j] = Z[i, 1] - Z[j, 1]

############################  Condition initiale  ############################

Uexa  =  Schemes2D.Sol_Exa_2D(cx, cy, D, X, Y, t)   # Solution exacte à l'instante t = 0
U     =  Uexa                        # Les deux solutions sont égaux à l'instante t = 0
u     =  U.reshape(N, 1)

#################  Calcule des matrices de différentiation  ##################

B, Bx, Bxx  =  Schemes2D.choose_method(method)(r, s, w, shape)
By, Byy     =  Schemes2D.choose_method(method)(r, w, s, shape)[1:3]
Bxy   = Bxx + Byy

Ax    =   np.dot(Bx, np.linalg.inv(B)) 
Ay    =   np.dot(By, np.linalg.inv(B))
Axy   =   np.dot(Bxy, np.linalg.inv(B))


###################  Affichage des solutions initiales  ######################

                      
# fig1 = plt.figure(1)
# ax1 = Axes3D(fig1)
# surf1 = ax1.plot_surface(X, Y, Uexa, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
# fig1.colorbar(surf1)
# plt.title('Solution approchée U en t = '+str(t))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


############  Calcule & affichage des solutions pour 0 < t <= T  #############


Nt  = int(T/dt) + 1

error_MAE  =  []
error_RMSE =  []
list = []
Err1 = 0


while t < T :
    
    dt    =  min(dt, T-t)                 # Pour calculer la dernière itération
    t     =  t + dt                       # Incrémentation du temps
    t     =  round(t, 4)
    list.append(t)
    
    ###########################  Solution Exacte  ############################
    
    Uexa  =  Schemes2D.Sol_Exa_2D(cx, cy, D, X, Y, t)
    
    uexa  =  Uexa.reshape(N, 1) 
            
    #########################  Solution Approchée  ###########################
                
    f = Schemes2D.ConvDiff2D           # f sous forme de l'équation d'advection-diffusion 2D
    
    u = Schemes2D.choose_schemes(schema)(dt, cx, cy, Ax, Ay, Axy, u, D, f)
    
    #########################  Conditions limites  ###########################
    
    for i in range(0, N) :
        if Z[i, 2] == 1 :
            u[i]  =  uexa[i]
    
    #####################  Solution Matrice Approchée  #######################
            
    U = u.reshape(Ny, Nx) 
      
    error_MAE.append(Schemes.MAE_error(uexa, u))      # Erreurs MAE pour la méthode MQ       
    error_RMSE.append(Schemes.RMSE_error(uexa, u))
    
    Err1 = Err1 + Schemes.RMSE_error(uexa, u)**2
    
    
    #####################  Affichage des deux solutions  #####################
    
    fig = plt.figure(figsize = plt.figaspect(0.4))
    plt.subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, U, rstride = 1, cstride = 1, cmap = 'coolwarm', edgecolor = 'none')
    plt.colorbar(surf, shrink = 0.8)
    ax1.set_title('Solution approchee U en t = '+str(t))
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

error_MAE_total  = max(error_MAE)
error_RMSE_total = np.sqrt(Err1/len(list))
print("error_MAE_total = ", "{:.3e}".format(error_MAE_total))
print("error_RMSE_total = ", "{:.3e}".format(error_RMSE_total))

################################  Errors Plot  ###############################

"""    
fig2  =  plt.figure(3)
ax    =  fig2.add_subplot(111)
plt.plot(list, error_MAE, 'r')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"L'evolution de l'erreur $MAE$ dans le temps")
plt.xlabel('t')
plt.ylabel(r'$MAE_{CS}}$ et $MAE_{MQ}$')
plt.legend([r"$MAE_{CS}$", r"$MAE_{MQ}$"])
plt.show()    
    

plt.close(plt.figure(3))  
plt.figure(3)
plt.plot(list, error_RMSE, 'r')
plt.title(r"L'évolution de l'erreur $MAE$ dans le temps")
plt.xlabel('t')
plt.ylabel(r'$MAE_{MQ}}$ et $MAE_{CS}$')
plt.legend([r"$MAE_{MQ}$", r"$MAE_{CS}$", r"$RMSE_{MQ}$", r"$RMSE_{CS}$"])
plt.show()  

plt.close(plt.figure(3))  
fig2  =  plt.figure(3)
ax    =  fig2.add_subplot(111)
plt.plot(list, error_MAE, 'r')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.title(r"L'evolution de l'erreur $MAE$ dans le temps")
plt.xlabel('t')
plt.ylabel(r'$MAE_{MQ}}$ et $MAE_{CS}$')
plt.legend([r"$MAE_{MQ}$"])#, r"$MAE_{CS}$", r"$RMSE_{MQ}$", r"$RMSE_{CS}$"])
plt.show()"""