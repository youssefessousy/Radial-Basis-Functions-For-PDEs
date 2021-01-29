"""
Created on Tue Jul 28 17:00:11 2020

@author: ESSOUSY YOUSSEF
"""

##############################################################################

""" 
    In this file, we call some functions from the file "Schemes.py", so make 
    sure the file "Schemes.py" exists in the same path where this file exists.
    
    In this program, we try to find numerical solutions for the Advecton-Diffusion
    equation with a discontinue initial condition. We are interested only in the 
    advection effect, so that's why we give numerical tests for weak values of 
    the diffusion term noted D. So this makes sense only when D is too small ~ 0.
    If it is not the case, the exact solution we gave will be no more a solution 
    for the advection-diffusion equation.
    
    The example we treat here using RBF method with Euler and Rung-Kutta's schemes
    for time stepping, is given in the project of Radial basis functions.
"""

import numpy as np
from matplotlib import pyplot as plt
import Schemes

###################  Schémas de discrétisation temporelle  ###################

schema  =  'Euler'                    # Schéma d'Euler
schema  =  'Rung-Kutta 2'             # Schéma de Rang-Kutta 2
schema  =  'Rung-Kutta 3'             # Schéma de Rang-Kutta 3
schema  =  'Rung-Kutta 4'             # Schéma de Rang-Kutta 4

######################  Choix de la fct RBF (MQ ou CS)  ######################

method  =  'CS-RBF'                   # Approximation par la fct MQ
method  =  'MQ-RBF'                   # Approximation par la fct CS

############################  Données initiales  #############################

a, b  =  -10,  10                     # [a, b] est le domaine spatiale
t, T  =  0, 5                         # Le temps initial et le temps final
Nx    =  300                          # Le nombre des points pris dans [a, b]
c     =  9                            # The shape parameter pour MQ
sigma =  10                           # The shape parameter pour CS
cr    =  1                            # La vitesse du champs
D     =  0.001                        # Le coefficient de diffusion
x     =  np.linspace(a, b, Nx)        # La subdivision uniforme de [a, b]
dx    =  (b-a)/(Nx-1)                 # Le pas de discétisation spatiale
cfl   =  0.95                         # La condition CFL de convergence pour le schéma "Rung-Kutta 4"
u     =  np.zeros((Nx, 1))            # Le vecteur Solution Approchée
uexa  =  np.zeros((Nx, 1))            # Le vecteur Solution Exacte
o     =  np.ones((1,len(x)))          # Vecteur sera utilisé pour calculer la matrices des distances (x_i - x_j)

###################  choix de shape parameter convenable  ####################

if method == 'MQ-RBF' :
    shape = c
elif method == 'CS-RBF' :
    shape = sigma
    
############################  Condition initiale  ############################

uexa  =  Schemes.Sol_Exact3(x, cr, t)   # Solution exacte à l'instante t= 0
u     =  uexa                           # Les deux solutions sont égaux à l'instante t = 0
     
##########################  Matrices des distances  ##########################

x    =   x.reshape(Nx,1)                              # Reshape(x) pour pouvoir calculer sa transpose
s    =   np.dot(x, o) - np.dot(x, o).transpose()      # La matrice (x_i - x_j)
r    =   abs(s)                                       # La matrice des distances |x_i - x_j|

##############################  Matrices RBF  ################################

B, Bx, Bxx  =  Schemes.choose_method(method)(r, s, shape)   # La matrice RBF et ces dérivée d'ordre 1 et 2
# On donne 'c' comme argument si la methode choisi est 'RBF-MQ', sinon, on donne 'sigma' pour la methode 'RBF-CS'.

Ax   =   np.dot(Bx, np.linalg.inv(B)) 
Axx  =   np.dot(Bxx, np.linalg.inv(B))

##################  Affichage des deux solutions à t = 0  ####################

plt.plot(x, u, 'r', x, u, 'b')
plt.grid()
plt.title(r'Les deux solutions $U_{app}$ et $U_{exa}$ en t = '+str("%.3f"%t))
plt.xlabel('x')
plt.ylabel(r'$U_{exa}}$ et $U_{app}$')
plt.legend([r"$U_{exa}(x,t)$", r"$U_{app}(x,t)$"])
plt.ylim(-0.3,5.3)
plt.time.sleep(0.01)
plt.show()

#############################  La condition CFL  #############################

dt_conv  =  cfl*dx/cr                   # Condition cfl pour le terme de transport
dt_diff  =  cfl*dx**2/(2*D)             # Condition cfl pour le terme de diffusion
dt       =  min(dt_conv, dt_diff)       # dt vérifie tous les deux conditions cfl
  
############  Calcule & affichage des solutions pour 0 < t <= T  #############
list1 = []
list2 = []

while t < T :
    
    dt   =  min(dt, T-t)                # Pour calculer la dernière itération
    t    =  t + dt                      # Incrémentation du temps
    t    =  round(t,4)
    list1. append(t)
    
    ###########################  Solution Exacte  ############################
    
    uexa = Schemes.Sol_Exact3(x, cr, t)
            
    #########################  Solution Approchée  ###########################
                
    f = Schemes.ConvDiff           # f sous forme de l'équation d'advection-diffusion
    
    u = Schemes.choose_schemes(schema)(cr, dt, Ax, Axx, u, D, f)
    
    ##################### Conditions limites (Direchlet) #####################
    
    # u[0]    =  uexa[0]
    # u[Nx-1] =  uexa[Nx-1]
    
    ###################### Conditions limites (Neumann) ######################
    
    u[0]    =  u[1]
    u[Nx-1] =  u[Nx-2]
    
    list2.append(Schemes.RMSE_error(uexa, u))
    
    #####################  Affichage des deux solutions  #####################
    
    plt.plot(x, u, 'r', x, uexa, 'b')
    plt.grid()
    plt.title(r'$U_{CS}$ et $U_{exa}$ en t = '+str("%.3f"%t))
    plt.xlabel('x')
    plt.ylabel(r'$U_{exa}}$ et $U_{cs}$')
    plt.legend([r"$U_{cs}(x,t)$", r"$U_{exa}(x,t)$"])
    #plt.ylim(-0.4,5.4)
    plt.time.sleep(0.01)
    plt.show()
    
##########################  Affichage des erreurs  ###########################
 
# plt.close(plt.figure(3))  
# plt.figure(3)
# plt.plot(list1, list2, 'r')
# plt.title(r"L'évolution de l'erreur $MAE$ dans le temps")
# plt.xlabel('t')
# plt.ylabel(r'$MAE_{MQ}}$')
# plt.legend([r"$MAE_{MQ}$"])
# plt.show()
