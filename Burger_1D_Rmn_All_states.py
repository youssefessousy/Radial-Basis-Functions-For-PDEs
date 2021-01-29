"""
Created on Mon Jul 27 02:33:48 2020

@author: ESSOUSY YOUSSEF
"""

##############################################################################

""" 
    In this file, we call some functions from the file "Schemes.py", so make
    sure the file "Schemes.py" exists in the same path where this file exists.
    
    In this program, we try to find numerical solutions for the Burger's
    equation with a discontinue initial condition "Riemann Problem". 
    The 
    Here We treat many examples of Two and Three-state Riemann problems given 
    in the project of Radial basis functions.
    
    We use RBF method (MQ and CS) with Euler and Rung-Kutta's schemes for time
    stepping.
    
    This makes sense only for weak values of the viscosity term noted D. otherwise,
    the exact solution we gave will be no more a solution for the Burger's equation.
    
    To choose which problem you want to solve (chock/Detente/Thee-state...),
    all you have is to comment and Uncomment the string variable named "Riemann
    _Prob".
"""

import numpy as np
from matplotlib import pyplot as plt
import Schemes

#######################  Choixs du problème de Riemann  ######################

Riemann_Prob  =  'Continue_State'
Riemann_Prob  =  'Three_States_Non_Monotone'
Riemann_Prob  =  'Three_States_Decreasing'
Riemann_Prob  =  'Three_States_Increasing'
Riemann_Prob  =  'Two_States_Choc'
Riemann_Prob  =  'Two_States_Detente'

###################  Schémas de discrétisation temporelle  ###################

schema  =  'Euler'                      # Schéma d'Euler
schema  =  'Rung-Kutta 2'               # Schéma de Rang-Kutta 2
schema  =  'Rung-Kutta 3'               # Schéma de Rang-Kutta 3
schema  =  'Rung-Kutta 4'               # Schéma de Rang-Kutta 4

######################  Choix de la fct RBF (MQ ou CS)  ######################

method  =  'CS-RBF'                     # Approximation par la fct MQ
method  =  'MQ-RBF'                     # Approximation par la fct CS

############################  Données initiales  #############################

a, b  =  -2, 2                        # [a, b] est le domaine spatiale
t, T  =  0, 5                         # Le temps initial et le temps final
Nx    =  300                          # Le nombre des points pris dans [a, b]
dt    =  0.005                        # Le pas du temps
c     =  30                           # The shape parameter pour MQ
sigma =  9                            # The shape parameter pour CS
D     =  0.0025                       # Le coefficient de viscositée
x     =  np.linspace(a, b, Nx)        # La subdivision uniforme de [a, b]
dx    =  (b-a)/(Nx-1)                 # Le pas de discétisation spatiale
cfl   =  0.3                          # La condition CFL de convergence pour le schéma "Rung-Kutta 4"
u     =  np.zeros((Nx, 1))            # Le vecteur Solution Approchée
uexa  =  np.zeros((Nx, 1))            # Le vecteur Solution Exacte
o     =  np.ones((1,len(x)))          # Vecteur sera utilisé pour calculer la matrices des distances (x_i - x_j) 

###################  choix de shape parameter convenable  ####################

if method == 'MQ-RBF' :
    shape = c
elif method == 'CS-RBF' :
    shape = sigma

############################  Condition initiale  ############################
                 
uexa  =  Schemes.choose_Riemann_Prob(Riemann_Prob)(x, t)     # Solution exacte à l'instante t= 0    
u  = uexa                             # Les deux solutions sont égaux à l'instante t = 0     

##########################  Matrices des distances  ##########################

x    =   x.reshape(Nx,1)                              # Reshape(x) pour pouvoir calculer sa transpose
s    =   np.dot(x, o) - np.dot(x, o).transpose()      # La matrice (x_i - x_j)
r    =   abs(s)                                       # La matrice des distances |x_i - x_j|

##############################  Matrices RBF  ################################

B, Bx, Bxx  =  Schemes.choose_method(method)(r, s, shape)   # La matrice RBF et ces dérivée d'ordre 1 et 2
# On donne 'c' comme argument si la methode choisi est 'RBF-MQ', sinon, o donne 'sigma' pour la methode 'RBF-CS'.

Ax   =   np.dot(Bx, np.linalg.inv(B)) 
Axx  =   np.dot(Bxx, np.linalg.inv(B))

##################  Affichage des deux solutions à t = 0  ####################

plt.plot(x, u, 'r', x, u, 'b')
plt.grid()
plt.title(r'Les deux solutions $U_{app}$ et $U_{exa}$ en t = '+str("%.3f"%t))
plt.xlabel('x')
plt.ylabel(r'$U_{exa}}$ et $U_{app}$')
plt.legend([r"$U_{exa}(x,t)$", r"$U_{app}(x,t)$"])
plt.ylim(-1.1,2.1)
plt.time.sleep(0.01)
plt.show()

############  Calcule & affichage des solutions pour 0 < t <= T  #############
Temps = []
error_MAE = []
error_RMSE = []

while t < T :
    dt  =  min(dt,T-t)
    t   =  t + dt
    Temps.append(t)
    
    ###########################  Solution Exacte  ############################
    
    uexa  =  Schemes.choose_Riemann_Prob(Riemann_Prob)(x, t)
           
    #########################  Solution Approchée  ###########################
    
    f = Schemes.BurgerEq            # f sous forme de l'équation de Burger 
         
    u = Schemes.choose_schemes(schema)(0, dt, Ax, Axx, u, D, f)  # Le 1er argument "0" ne joue aucun role : Arg Optionel
    
    #########################  Conditions limites  ###########################
    
    u[0]    =  uexa[0]
    u[Nx-1] =  uexa[Nx-1]
    
    #####################  Affichage des deux solutions  ##################### 
    
    plt.plot(x, uexa, 'r', x, u, 'b', mfc = 'none', linewidth = 0.75)
    plt.grid()
    plt.title(r'Les deux solutions $U_{app}$ et $U_{exa}$ en t = '+str("%.3f"%t))
    plt.xlabel('x')
    plt.ylabel(r'$U_{exa}}$ et $U_{app}$')
    plt.legend([r"$U_{exa}(x,t)$", r"$U_{app}(x,t)$"])
    #plt.ylim(-1.1,2.1)
    plt.time.sleep(0.01)
    plt.show()