# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:33:53 2020

@author: ESSOUSY YOUSSEF
"""

import numpy as np


######################  Solution exacte de Shallow Water #####################

def Sol_Shallow_Water(x, y, t):
    nu0 = 1
    w = 1.45444*0.12
    # l = 50                # Not needed for the analytical solution.
    L = 872
    H = 20
    g = 9.81
    nu = nu0*np.cos(w*(L - x)/np.sqrt(g*H))*np.cos(w*t)/np.cos(w*L/np.sqrt(g*H))
    u  = -nu0*np.sqrt(g/H)*np.sin(w*(L - x)/np.sqrt(g*H))*np.sin(w*t)/np.cos(w*L/np.sqrt(g*H))
    v  = 0
    return nu, u, v


def F(nu, u, v, Ax, Ay) :
    H = 20
    g = 9.81
    a = -H*(np.dot(Ax, u) + np.dot(Ay, v))
    b = -g*np.dot(Ax, nu)
    c = -g*np.dot(Ay, nu)
    return a, b, c

def Time_Scheme(nu, u, v, Ax, Ay, dt, f) :
    k1, k2, k3  =  f(nu, u, v, Ax, Ay)
    h1   =  nu + dt*k1
    h2   =  u + dt*k2
    h3   =  v + dt*k3
    m1, m2, m3  =  f(h1, h2, h3, Ax, Ay)
    n1   =  (3/4)*nu + (1/4)*h1 + (dt/4)*m1
    n2   =  (3/4)*u + (1/4)*h2 + (dt/4)*m2
    n3   =  (3/4)*v + (1/4)*h3 + (dt/4)*m3
    p1, p2, p3  =  f(n1, n2, n3, Ax, Ay)
    w1   =  (1/3)*nu + (2/3)*n1 + (2*dt/3)*p1
    w2   =  (1/3)*u + (2/3)*n2 + (2*dt/3)*p2
    w3   =  (1/3)*v + (2/3)*n3 + (2*dt/3)*p3
    return w1, w2, w3
    
    