"""Created on Sun May 31 18:10:31 2020

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

def ConvDiff(cr, Ax, Axx, u, D) : 
    return -cr*np.dot(Ax, u) + D*np.dot(Axx, u)

###############  L'équation de Burger sous forme vectorielle  ################

def BurgerEq(cr, Ax, Axx, u, D) : 
    return -(1/2)*np.dot(Ax, u**2) + D*np.dot(Axx, u)
    #return -u*np.dot(Ax, u) + D*np.dot(Axx, u)

#############################  Schéma d'Euler  ###############################
    
def Euler(cr, dt, Ax, Axx, u, D, f) :
    K1  = f(cr, Ax, Axx, u, D) 
    return u + dt*K1
    
#########################  Schéma de Rang-Kutta 2  ###########################
    
def RK2(cr, dt, Ax, Axx, u, D, f) :
    K1  = u + dt*f(cr, Ax, Axx, u, D) 
    K2  = f(cr, Ax, Axx, K1, D) 
    return u + dt*K2 

#########################  Schéma de Rang-Kutta 3  ###########################
    
def RK3(cr, dt, Ax, Axx, u, D, f) :
     K1 = u + dt*f(cr, Ax, Axx, u, D)
     K2 = 3*u/4 + K1/4 + dt*f(cr, Ax, Axx, K1, D)/4
     return u/3 + 2*K2/3 + 2*dt*f(cr, Ax, Axx, K2, D)/3

#########################  Schéma de Rang-Kutta 4  ###########################
     
def RK4(cr, dt, Ax, Axx, u, D, f) :
    K1  = dt*f(cr, Ax, Axx, u, D) 
    aux = u + 0.5*K1 
    K2  = dt*f(cr, Ax, Axx, aux, D) 
    aux = u + 0.5*K2 
    K3  = dt*f(cr, Ax, Axx, aux, D) 
    aux = u + K3 
    K4  = dt*f(cr, Ax, Axx, aux, D) 
    return u + (K1 + 2*K2 + 2*K3 + K4)/6 

#####################  Les matrices  CS et ces dérivées  #####################

def MQ_Derivatives(r, s, c) :
    k  =  np.sqrt(1 + (c*r)**2)               # La matrice MQ      
    m  =  (c**2)*s/np.sqrt(1 + (c*r)**2)      # La dérivée d'ordre 1 MQ
    n  =  (c**2)/(1 + (c*r)**2)**(3/2)        # La dérivée d'ordre 2 MQ
    return k, m, n

####################  La fonction CS et ces dérivées  ########################
 
def CSRBF(x, s, c) :                 # x est supposé positive
    if x/c <= 1 :                    # c la constante qui controle le support
        k = (1 - x/c)**6 *(3 + 18*x/c + 35*(x/c)**2)    # La fonction CS
        m = -(56/c**2) * s * (1 - x/c)**5 * (1 + 5*x/c)  #La dérivé 1er
        n = -(56/c**2) * (1 - x/c)**4 * (1 + 4*x/c - 35*(x/c)**2) #La dérivée 2em
    else :
        k = 0
        m = 0
        n = 0
    return k, m, n

####################  Les matrices  CS et ces dérivées  ######################

def CS_Derivatives(r, s, c) :            # s(i,j) = x_i - x_j
    k = np.zeros((len(r), len(r[0])))    # La matrice CS
    m = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 1 CS
    n = np.zeros((len(r), len(r[0])))    # La dérivé d'ordre 2 CS
    for i in range(0, len(r)) :
        for j in range(0, len(r[0])) :
            k[i, j], m[i, j], n[i, j] = CSRBF(r[i, j], s[i, j], c)   
    return k, m, n

####################  Choix des schémas d'approximation  #####################
    
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
        f  =  MQ_Derivatives
    elif method == 'CS-RBF' :
        f  =  CS_Derivatives 
    return f

##################  Solution Exacte d'eq de Burger continue  #################
    
def Sol_Continue(x, t) :
    u  =  np.zeros((len(x), 1))
    for i in range(0, len(x)) :
        u[i] = x[i]/(t + 1)
    return u

###########  Solution Exacte d'eq de Burger 2-States Croissante  ############
    
def Sol_2states_detente(x, t) :
    u  =  np.zeros((len(x), 1))
    for i in range(0, len(x)) :
        if x[i] <= 0 :
            u[i] = 0
        elif 0 < x[i] and x[i] <= t :
            u[i] = x[i]/t
        else :
            u[i] = 1
    return u

############  Solution Exacte d'eq de Burger 2-States Décroissante  ##########
    
def Sol_2states_choc(x, t) :
    u  =  np.zeros((len(x), 1))
    for i in range(0, len(x)) :
        if x[i] <= t/2 :
            u[i] = 1
        else :
            u[i] = 0
    return u

############  Solution Exacte d'eq de Burger 3-States Croissante  ############

def Sol_3states_increasing(x, t) :
    u  =  np.zeros((len(x), 1))
    for i in range(0, len(x)) :
        if x[i] <= -t :
            u[i] = -1
        elif -t < x[i] and x[i] <= 0 :
            u[i] = x[i]/t
        elif 0 < x[i] and x[i] <= 1 :
            u[i] = 0
        elif 1 < x[i] and x[i] <= 2*t+1 :
            u[i] = (x[i]-1)/t
        else :
            u[i] = 2
    return u

############  Solution Exacte d'eq de Burger 3-States Déroissante  ###########
    
def Sol_3states_decreasing(x, t) :
    u  =  np.zeros((len(x), 1))
    if t <= 2/3 :
        for i in range(0, len(x)) :
            if x[i] <= t :
                u[i] = 2
            elif t < x[i] and x[i] <= -t/2 + 1 :
                u[i] = 0
            else :
                u[i] = -1
    else :
        for i in range(0, len(x)) :
            if x[i] <= t/2 + 1/3 :
                u[i] = 2
            else :
                u[i] = -1
    return u

############  Solution Exacte d'eq de Burger 3-States Non Monotone  ##########
    
def Sol_3states_Non_Monotone(x, t) :
    u  =  np.zeros((len(x), 1))
    if t <= 1 :
        for i in range(0, len(x)) :
            if x[i] <= 0 :
                u[i] = 0
            elif 0 < x[i] and x[i] <= t :
                u[i] = x[i]/t
            elif t < x[i] and x[i] <= 1:
                u[i] = 1
            else :
                u[i] = -1
    elif 1 < t and t < 4 :
        for i in range(0, len(x)) :
            if x[i] <= 0 :
                u[i] = 0
            elif 0 < x[i] and x[i] <= 2*np.sqrt(t) - t :
                u[i] = x[i]/t
            else :
                u[i] = -1
    else :
        for i in range(0, len(x)) :
            if x[i] <= -t/2 + 2 :
                u[i] = 0
            else :
                u[i] = -1
    return u

######################  Choix du problème de Riemann  ########################
    
def choose_Riemann_Prob(Riemann_Prob) :
    if   Riemann_Prob == 'Continue_State' :
        f = Sol_Continue
    if   Riemann_Prob == 'Two_States_Detente' :
        f = Sol_2states_detente
    elif Riemann_Prob == 'Two_States_Choc' :
        f = Sol_2states_choc
    elif Riemann_Prob == 'Three_States_Increasing' :
        f = Sol_3states_increasing
    elif Riemann_Prob == 'Three_States_Decreasing' :
        f = Sol_3states_decreasing
    elif Riemann_Prob == 'Three_States_Non_Monotone' :
        f = Sol_3states_Non_Monotone
    return f

####################  Solution Exacte d'eq d'advection  ######################

def Sol_Exact3(x, c, t) :
    u  =  np.zeros((len(x), 1))
    for i in range(0, len(x)) :
        if x[i]-c*t >= -1 and x[i]-c*t <= 1 :
            u[i] = 5
        else :
            u[i] = 0
    return u

############################  Erreur Absolu MAE  #############################
    
def MAE_error(u, v) :
    w = abs(u - v)
    return float(max(w))

#####################  Erreur Quadratique Moyenne RMSE  ######################

def RMSE_error(u, v) :
    w = 0
    for i in range(0, len(u)) :
        w = w + (u[i] - v[i])**2
    return float(np.sqrt(w/len(u)))
