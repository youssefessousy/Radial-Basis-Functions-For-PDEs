"""
Created on Fri Jul 31 01:23:15 2020

@author: ESSOUSY YOUSSEF
"""

##############################################################################

"""
    This file contains the functions and the schemes we need to run other       
    programs to find numerical solutions for the advection and the Burger's 
    equation. All these functions are imported in other files, so make sure 
    these files exist in the same path where this file "Schemes.py" exists.
"""

import numpy as np

#########  L'équation d'advection-Diffusion sous forme vectorielle  ##########

def ConvDiff2D(cx, cy, Ax, Ay, Axy, u, D) : 
    return -cx*np.dot(Ax, u) - cy*np.dot(Ay, u) + D*np.dot(Axy, u)

##############  L'équation de Burger 2D sous forme vectorielle  ##############

def Burger2D(cx, cy, Ax, Ay, Axy, u, D) : 
    #return -u*np.dot(Ax, u) - u*np.dot(Ay, u) + D*np.dot(Axy, u)
    return -0.5*np.dot(Ax, u**2) - 0.5*np.dot(Ay, u**2) + D*np.dot(Axy, u)

##############  L'équation de Burger 2D (sans diffusion)  ##############
    
def Burger2D_RMN(cx, cy, Ax, Ay, Axy, u, D) : 
    #return -u*np.dot(Ax, u) - u*np.dot(Ay, u) + D*np.dot(Axy, u)
    return -0.5*np.dot(Ax, u**2) - 0.5*np.dot(Ay, u**2)

#############################  Schéma d'Euler  ###############################
    
def Euler(dt, cx, cy, Ax, Ay, Axy, u, D, f) :
    K1  = f(cx, cy, Ax, Ay, Axy, u, D) 
    return u + dt*K1
    
#########################  Schéma de Rang-Kutta 2  ###########################
    
def RK2(dt, cx, cy, Ax, Ay, Axy, u, D, f) :
    K1  = u + dt*f(cx, cy, Ax, Ay, Axy, u, D) 
    K2  = f(cx, cy, Ax, Ay, Axy, K1, D) 
    return u + dt*K2 

#########################  Schéma de Rang-Kutta 3  ###########################
    
def RK3(dt, cx, cy, Ax, Ay, Axy, u, D, f) :
     K1 = u + dt*f(cx, cy, Ax, Ay, Axy, u, D)
     K2 = 3*u/4 + K1/4 + dt*f(cx, cy, Ax, Ay, Axy, K1, D)/4
     return u/3 + 2*K2/3 + 2*dt*f(cx, cy, Ax, Ay, Axy, K2, D)/3

#########################  Schéma de Rang-Kutta 4  ###########################
     
def RK4(dt, cx, cy, Ax, Ay, Axy, u, D, f) :
    K1  = dt*f(cx, cy, Ax, Ay, Axy, u, D)
    aux = u + 0.5*K1 
    K2  = dt*f(cx, cy, Ax, Ay, Axy, aux, D)
    aux = u + 0.5*K2 
    K3  = dt*f(cx, cy, Ax, Ay, Axy, aux, D) 
    aux = u + K3 
    K4  = dt*f(cx, cy, Ax, Ay, Axy, aux, D)
    return u + (K1 + 2*K2 + 2*K3 + K4)/6 

####################  La matrice MQ et ces dérivées en 2D  ###################

def MQ_Derivatives2D(r, s, w, c) :   # s(i,j) = x_i - x_j et # w(i,j) = y_i - y_j
    k  =  np.sqrt(1 + (c*r)**2)                              # La matrice MQ      
    m  =  (c**2 * s)/np.sqrt(1 + (c*r)**2)                   # La dérivée d'ordre 1 MQ % x
    n  =  c**2 * (1 + (c*w)**2)/(1 + (c*r)**2)**(3/2)    # La dérivée d'ordre 2 MQ % x
    return k, m, n

""" Les deux dérivées calculées içi sont par rapport à x (s). Alors pour calculer
 les dérivées par rapport à y, il suffit de permuter s et w lors de l'appel de
 la fonction MQ_Derivatives2D."""

####################  La fonction CS et ces dérivées  ########################
 
def CSRBF2D(x, s, c) :               # x est supposé positive
    if x/c <= 1 :                     # c la constante qui controle le support
        k = (1 - x/c)**6 *(3 + 18*x/c + 35*(x/c)**2)     # La fonction CS
        m = -(56/c**2) * s * (1 - x/c)**5 * (1 + 5*x/c)  # La dérivé 1er
        n = -(56/c**2) * (1 - x/c)**4 * (1 + 4*x/c - 5*(x/c)**2 - 30*(s/c)**2) #La dérivée 2em
    else :
        k = 0
        m = 0
        n = 0
    return k, m, n       

######################  La matrice CS et ces dérivées  #######################

def CS_Derivatives2D(r, s, w, c) :       # s(i,j) = x_i - x_j et w(i,j) = y_i - y_j 
    k = np.zeros((len(r), len(r[0])))    # La matrice CS
    m = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 1 CS
    n = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 2 CS
    for i in range(0, len(r)) :
        for j in range(0, len(r[0])) :
            k[i, j], m[i, j], n[i, j] = CSRBF2D(r[i, j], s[i, j], c)   
    return k, m, n

""" Remarque : On a pas besoin d'argument w, mais on le garde parce que on aura 
 besoin que la fonction CS_Derivatives porte le même nombre d'arguments que la
 la fonction MQ_Derivatives dans la méthode choose_schemes.
 Pour calculer les dérivées par rapport à y, il suffit de permuter s et w lors 
 de l'appel de la fonction CS_Derivatives2D."""

####################  La matrice MQ et ces dérivées en 3D  ###################

def MQ_Derivatives3D(r, s, w, v, c) :   # s(i,j) = x_i - x_j et # w(i,j) = y_i - y_j
                                        # v(i,j) = z_i - z_j
    k  =  np.sqrt(1 + (c*r)**2)                              # La matrice MQ      
    m  =  (c**2 * s)/np.sqrt(1 + (c*r)**2)                   # La dérivée d'ordre 1 MQ % x
    n  =  c**2 * (1 + c**2*(w**2 + v**2))/(1 + (c*r)**2)**(3/2)  # La dérivée d'ordre 2 MQ % x
    return k, m, n

""" Les deux dérivées calculées içi sont par rapport à x (s). Alors pour calculer
 les dérivées par rapport à y ou z, il suffit de permuter s et w ou v lors de 
 l'appel de la fonction MQ_Derivatives2D."""

###################  La matrice CS et ces dérivées en 3D  ####################

def CS_Derivatives3D(r, s, w, v, c) :    # s(i,j) = x_i - x_j et # w(i,j) = y_i - y_j
                                         # v(i,j) = z_i - z_j 
    k = np.zeros((len(r), len(r[0])))    # La matrice CS
    m = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 1 CS
    n = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 2 CS
    for i in range(0, len(r)) :
        for j in range(0, len(r[0])) :
            k[i, j], m[i, j], n[i, j] = CSRBF2D(r[i, j], s[i, j], c)   
    return k, m, n

""" Remarque : On a pas besoin d'argument w, mais on le garde parce que on aura 
 besoin que la fonction CS_Derivatives porte le même nombre d'arguments que la
 la fonction MQ_Derivatives dans la méthode choose_schemes.
 Pour calculer les dérivées par rapport à y, il suffit de permuter s et w lors 
 de l'appel de la fonction CS_Derivatives2D."""
 
# ########################  La matrice multiquadrique  #########################
    
# def MQ(r, c) :
#     return np.sqrt(c**2 + r**2)

# ########################  Les matrices dérivées MQ  ##########################

# def MQ_Derivatives(r, s, w, c, d) :
#     if d == 1 :
#         m = (c**2)*(s + w)/np.sqrt(1 + (c*r)**2)            # Divergence MQ
#     elif d == 2 :
#         m = (c**2)*(2 + (c*r)**2)/(1 + (c*r)**2)**(3/2)      # Laplacien MQ
#     return m

# ####################  La fonction CS et ces dérivées  ########################
 
# def CSRBF(x, c) :                    # x est supposé positive
#     if x/c <= 1 :                    # c la constante qui controle le support
#         m = (1 - x/c)**6 *(3 + 18*x/c + 35*(x/c)**2)
#     else :
#         m = 0
#     return m

# ########################  La Dérivée première CS  ############################
    
# def CS_Diverg(x, s, w, c) :                    # s(i,j) = x_i - x_j
#     if x/c <= 1 :                         # x(i,j) = |x_i - x_j| = |s(i,j)|
#         m = -(56/c**2) * (s + w) * (1 - x/c)**5 * (1 + 5*x/c)
#     else :
#         m = 0
#     return m

# ########################  La Dérivée seconde CS  #############################

# def CS_Laplacien(x, s, w, c) :
#     if x/c <= 1 :
#         m = -(56/c**2) * (1 - x/c)**4 * (1 + 4*x/c - 40*(x/c)**2)
#     else :
#         m = 0
#     return m

# #####################  La matrice compactly supported  #######################
    
# def CS(r, c) :
#     m = np.zeros((len(r), len(r[0])))
#     for i in range(0, len(r)) :
#         for j in range(0, len(r[0])) :
#             m[i, j] = CSRBF(r[i, j], c)
#     return m

# ########################  Les matrices dérivées CS  ##########################

# def CS_Derivatives(r, s, w, c, d) :            # s(i,j) = x_i - x_j
#     m = np.zeros((len(r), len(r[0])))       # x(i,j) = |x_i - x_j| = |s(i,j)|
#     if d == 1 :                                      # Dérivée d'ordre 1
#         for i in range(0, len(r)) :
#             for j in range(0, len(r[0])) :
#                 m[i, j] = CS_Diverg(r[i, j], s[i, j], w[i, j], c)
#     elif d == 2 :                                    # Dérivée d'ordre 2
#         for i in range(0, len(r)) :
#             for j in range(0, len(r[0])) :
#                 m[i, j] = CS_Laplacien(r[i, j], s[i, j], w[i, j], c)       
#     return m

#######################  Les schémas d'approximation  ########################
    
def choose_schemes(schema) :
    if   schema == 'Euler' :
        f = Euler
    elif schema == 'Rung-Kutta 2' :
        f = RK2
    elif schema == 'Rung-Kutta 3' :
        f = RK3
    elif schema == 'Rung-Kutta 4' :
        f = RK4
    return f

#################  Choix ds fonctions d'approximation RBF  ###################

def choose_method(method) :
    if   method == 'MQ-RBF' :
        f  =  MQ_Derivatives2D
    elif method == 'CS-RBF' :
        f  =  CS_Derivatives2D 
    return f

#################  Choix ds fonctions d'approximation RBF  ###################

def choose_method_3D(method) :
    if   method == 'MQ-RBF' :
        f  =  MQ_Derivatives3D
    elif method == 'CS-RBF' :
        f  =  CS_Derivatives3D 
    return f

#####################  Solution Exacte d'eq d'Adv-Dif 2D  ####################

def Sol_Exa_2D(cx, cy, D, x, y, t) :
    K1 = -(x - cx*t)**2/(D*(4*t + 1))
    K2 = -(y - cy*t)**2/(D*(4*t + 1))
    return np.exp(K1 + K2)/(4*t + 1) 
    
    
    
    