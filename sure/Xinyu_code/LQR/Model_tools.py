# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: some matrix used in the model

from initialization import *
import numpy as np
from numpy import linalg as la
from EigenTest import M_eval

def D_matrix(G1,G2):
    '''matrix D'''
    M11        = a11
    M12        = a12
    M21        = a21+G1
    M22        = a22+G2

    D   = np.array([[M11**2,  2*M11*M12,        M12**2 ],
                    [M11*M21, M11*M22+M12*M21,  M22*M12],
                    [M21**2,  2*M21*M22,        M22**2 ]])
    return D

def S_matrix(G1,G2,returnXiS,display):
    '''a function used to compute S theoretically
    @parameter: if returnXiS=True, return xi_S (3*1); else return S (2*2)'''
    D = D_matrix(G1,G2)
    xi_S = la.inv(np.eye(3)-D)@xi_R
    if returnXiS:
        return xi_S
    
    S = np.zeros([2,2])
    S[0,0] = xi_S[0,0]
    S[1,0] = xi_S[1,0]
    S[0,1] = xi_S[1,0]
    S[1,1] = xi_S[2,0]
    
    if display:
        print("Covariance matrix S (theoretically):\n",S)
        G = np.array([[G1,G2]])
        M = A+B@G
        print("MSM.T+R:\n",M@S@M.T+R)
    return S

def D_dot(G1,G2,returnD1):
    '''The derivative of D w.r.t G1 or G2
    @parameter: returnD1==True, return D_dot w.r.t G1; else, return G2'''
    D_1 = np.array([[0,          0,                  0],
                    [a11,        a12,                0],
                    [2*(a21+G1), 2*(a22+G2),         0]])
    if returnD1:
        return D_1
    
    D_2 = np.array([[0,          0,                  0],
                    [0,          a11,              a12],
                    [0,          2*(a21+G1), 2*(a22+G2)]])
    return D_2

def M_matrix(G1,G2):
    G = np.array([[G1,G2]])
    return A+B@G

'''
#SEEMS HAVE BUG
def S_dot2(G1,G2,returnS1,returnXi,display):
    #The derivative of S w.r.t G1 or G2
    #@parameter: returnS1==True, return S_dot w.r.t G1; else, return G2
    #            returnXi==True, return xiS_dot (3*1); else, return S_dot(2*2)
    D = D_dot(G1,G2,returnS1)         # according to the formula in write up, we need to calculate D_dot first
    E = la.inv(np.eye(3)-D)           # (I-D)^(-1)
    D_Dot = D_dot(G1,G2,returnS1)     # find out D_dot
    xi_S_dot = E@D_Dot@E@xi_R           
    if returnXi:
        return xi_S_dot
    S_Dot = np.zeros([2,2])
    S_Dot[0,0] = xi_S_dot[0,0]
    S_Dot[1,0] = xi_S_dot[1,0]
    S_Dot[0,1] = xi_S_dot[1,0]
    S_Dot[1,1] = xi_S_dot[2,0]
    if display:
        print("S dot w.r.t G%d (theoretically):"%(2-returnS1))
        print(S_Dot) 
    return S_Dot
'''

def M_dot(returnM1):
    '''differentiate M w.r.t G
    @parameter: returnM1 == True if we need to differentiate M w.r.t G1'''
    M = np.zeros([2,2])
    if returnM1:
        M[1,0] = 1
        return M
    M[1,1] = 1
    return M

def u_matrix(G1,G2,returnU1,returnXi):
    '''u=dM S M^T + M S dM^T
    @parameter: returnU1 == True if we need to diffrentiate w.r.t G1
                returnxiU== True if we need 3*1 xi_U'''
    dM = M_dot(returnU1)
    M  = M_matrix(G1,G2)
    S  = S_matrix(G1,G2,returnXiS=False,display=False)
    u  = dM@S@M.T+M@S@dM.T
    if returnXi:
        xiU = np.array([[u[0,0],u[1,0],u[1,1]]]).T
        return xiU
    return u
    
def S_dot(G1,G2,returnS1,returnXi,display):
    xiU = u_matrix(G1,G2,returnS1,returnXi=True)
    D   = D_matrix(G1,G2)
    xiS_dot = la.inv((np.eye(3)-D))@xiU 
    if returnXi:
        return xiS_dot
    dS  = np.array([[xiS_dot[0,0],xiS_dot[1,0]],[xiS_dot[1,0],xiS_dot[2,0]]])
    if display:
        print("S dot w.r.t G%d (theoretically):"%(2-returnS1))
        print(dS)
    return dS
    
def H_matrix(G1,G2):
    return np.array([[G1**2, 2*G1*G2, G2**2]])

