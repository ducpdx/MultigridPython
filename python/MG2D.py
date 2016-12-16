
# coding: utf-8

# In[1]:

'''
import Python libraries and presets
'''
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../FigStyle/plot.mplstyle')


# In[2]:

'''
ROUNTINES FOR DATASTRUCTURE
'''
def initialize(nx0, ny0, maxlevel, xa, xb, ya, yb):
    ii    = np.array([nx0*(2**i) for i in range(maxlevel)])
    jj    = np.array([ny0*(2**i) for i in range(maxlevel)])    
    hx    = (xb-xa)*1./ii 
    hy    = (yb-ya)*1./jj
    f     = np.array([np.zeros((ii[k]+1, jj[k]+1)) for k in range(maxlevel)], 
             dtype=np.ndarray)
    u     = np.array([np.zeros((jj[k]+1, ii[k]+1)) for k in range(maxlevel)], 
             dtype=np.ndarray)
    uold  = np.array([np.zeros((ii[k]+1, jj[k]+1)) for k in range(maxlevel)], 
             dtype=np.ndarray)
    uconv = np.array([np.zeros((ii[k]+1, jj[k]+1)) for k in range(maxlevel)], 
             dtype=np.ndarray)
    return ii,jj,hx,hy,f,u,uold,uconv


# In[3]:

'''
FUNCTIONS
'''
def U_a(x,y):
    #analytical solution
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def U_b(x,y):
    #take analytical solution as boundary condition
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def U_i(x,y):
    #initial approximation
    return 0.0
def F_i(x,y,EPSILON):
    #right hand side function
    return -4*(1+EPSILON)*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def Lu(u, rhx2, rhy2, i, j, EPSILON): 
    #compute operator in point i,j  
    return(rhx2*(u[i-1][j  ]-2*u[i][j]+u[i+1][j  ])          +rhy2*(u[i  ][j-1]-2*EPSILON*u[i][j]+u[i  ][j+1]))


# In[4]:

'''
SINGLE GRID ROUTINES
'''
def init_uf(u, f, k, ii, jj, xa, ya, hx, hy, EPSILON):
    '''
    initialize u, sets boundary condition for u, and 
    initialize right hand side values on grid level
    '''
    x = 0.0
    y = 0.0
    for i in range(ii[k]):
        x = xa + i*hx[k]
        for j in range(jj[k]):
            y = ya + j*hx[k]
            if ((i==0) or (j==0) or (i==ii[k]) or (j==jj[k])):
                u[k][i][j]=U_b(x,y)
            else:
                u[k][i][j]=U_i(x,y)
            f[k][i][j] = F_i(x,y,EPSILON)
    return u,f

def relax(u, f, k, ii, jj, hx, hy, wu, EPSILON):
    #perform Gauss-Seidel relaxation on gridlevel k 
    rhx2 = 1.0/(hx[k]*hx[k]); 
    rhy2 = 1.0/(hy[k]*hy[k]);
    
    for i in range (1,ii[k]): 
        for j in range (1,jj[k]): 
            u[k][i][j] += (f[k][i][j]-Lu(u[k],rhx2,rhy2,i,j, EPSILON))/(-2*rhx2-2*rhy2)
  
    err = 0.0 
    for i in range (1,ii[k]): 
        for j in range (1,jj[k]): 
            err += np.fabs(f[k][i][j]-Lu(u[k],rhx2,rhy2,i,j,EPSILON))

    wu += np.exp((maxlevel-k-1)*np.log(.25))
    print("\nLevel %d Residual %8.5e Wu %8.5e"%(k+1, err/((ii[k]-1)*(jj[k]-1)),wu))
    return u, wu

def conver_a(u, k, ii, jj, xa, ya, hx, hy): 
    '''
    compute L1 norm of difference between U and analytic solution 
    on level k
    '''
    
    err = 0.0
    
    for i in range (1,ii[k]): 
        x = xa + i*hx[k]
        for j in range (1,jj[k]):
            y = ya + j*hx[k]
            err += np.fabs(u[k][i][j]-U_a(x,y))
    
    return (err/((ii[k]-1)*(jj[k]-1)))


# In[5]:

'''
INTER GRID ROUTINES
'''
def coarsen_u(u, uold, k, ii, jj, hx, hy, wu, EPSILON):
    '''
    compute initial approximation on level k-1 
    in coarsening step from level k
    '''
    
    iic = ii[k-1]
    jjc = jj[k-1]
    
    for ic in range (1,iic): 
        for jc in range (1,jjc):
            u[k-1][ic][jc] = .0625*(     u[k][2*ic-1][2*jc-1]+u[k][2*ic-1][2*jc+1]                                        +u[k][2*ic+1][2*jc-1]+u[k][2*ic+1][2*jc+1]                                   +2.0*(u[k][2*ic-1][2*jc  ]+u[k][2*ic+1][2*jc  ]                                   +     u[k][2*ic  ][2*jc-1]+u[k][2*ic  ][2*jc+1])                                   + 4.0*u[k][2*ic  ][2*jc  ])

    #store coarse grid solution in uco array
    for ic in range (iic+1): 
        for jc in range (jjc+1):
            uold[k-1][ic][jc] = u[k-1][ic][jc]  
    return u,uold

def coarsen_f(u, f, k, ii, jj, hx, hy, wu, EPSILON):
    '''
    compute coarse grid right hand side on level k-1 
    in coarsening step from level k
    '''
    
    iic = ii[k-1]
    jjc = jj[k-1]
    
    rh2xc = 1.0/(hx[k-1]*hx[k-1])
    rh2yc = 1.0/(hy[k-1]*hy[k-1])
    
    rh2x  = 1.0/(hx[k]*hx[k]) 
    rh2y  = 1.0/(hy[k]*hy[k])
    
    for ic in range (1,iic): 
        for jc in range (1,jjc):
            rc  = (f[k][2*ic  ][2*jc  ] - Lu(u[k],rh2x,rh2y,2*ic  ,2*jc  ,EPSILON))
            rn  = (f[k][2*ic  ][2*jc+1] - Lu(u[k],rh2x,rh2y,2*ic  ,2*jc+1,EPSILON))
            re  = (f[k][2*ic+1][2*jc  ] - Lu(u[k],rh2x,rh2y,2*ic+1,2*jc  ,EPSILON))
            rs  = (f[k][2*ic  ][2*jc-1] - Lu(u[k],rh2x,rh2y,2*ic  ,2*jc-1,EPSILON))
            rw  = (f[k][2*ic-1][2*jc  ] - Lu(u[k],rh2x,rh2y,2*ic-1,2*jc  ,EPSILON))
            rne = (f[k][2*ic+1][2*jc+1] - Lu(u[k],rh2x,rh2y,2*ic+1,2*jc+1,EPSILON))
            rse = (f[k][2*ic+1][2*jc-1] - Lu(u[k],rh2x,rh2y,2*ic+1,2*jc-1,EPSILON))
            rsw = (f[k][2*ic-1][2*jc-1] - Lu(u[k],rh2x,rh2y,2*ic-1,2*jc-1,EPSILON))
            rnw = (f[k][2*ic-1][2*jc+1] - Lu(u[k],rh2x,rh2y,2*ic-1,2*jc+1,EPSILON))
            #FAS coarse grid right hand sie   
            #with full weighting of residuals 
            f[k-1][ic][jc] = Lu(u[k-1],rh2xc,rh2yc,ic,jc,EPSILON)                                 +.0625*(rne+rse+rsw+rnw+2.0*(rn+re+rs+rw)+4.0*rc)    
    return f

def refine(u, uold, f, k, ii, jj, hx, hy, EPSILON):
    '''
    Interpolation and addition of coarse grid correction from grid k-1 
    to grid k   
    '''
    iic = ii[k-1]
    jjc = jj[k-1]

    for ic in range (1,iic+1): 
        for jc in range (1,jjc+1):
            if (ic<iic): 
                u[k][2*ic  ][2*jc  ] += (u[k-1][ic  ][jc]-uold[k-1][ic][jc])

            if (jc<jjc): 
                u[k][2*ic-1][2*jc  ] += (u[k-1][ic  ][jc]-uold[k-1][ic  ][jc]                                        +u[k-1][ic-1][jc]-uold[k-1][ic-1][jc])*0.5

            if (ic<iic):
                u[k][2*ic  ][2*jc-1] += (u[k-1][ic  ][jc  ]-uold[k-1][ic  ][jc  ]                                        +u[k-1][ic][jc-1  ]-uold[k-1][ic  ][jc-1])*0.5
                
            u[k][2*ic-1][2*jc-1]     += (u[k-1][ic  ][jc  ]-uold[k-1][ic  ][jc  ]                                        +u[k-1][ic][jc-1  ]-uold[k-1][ic  ][jc-1]                                        +u[k-1][ic-1][jc  ]-uold[k-1][ic-1][jc  ]                                        +u[k-1][ic-1][jc-1]-uold[k-1][ic-1][jc-1])*0.25
    return u

def fmg_interpolate(u, uconv, f, k, ii, jj, hx, hy, EPSILON):
    '''
    interpolation of coarse grid k-1 solution to fine grid 
    to serve as first approximation bi-cubic interpolation 
    '''
    iic = ii[k-1]
    jjc = jj[k-1]

    #store grid k-1 solution for later use in convergence check 

    for ic in range (1,iic): 
        for jc in range (1,jjc):
            uconv[k-1][ic][jc] = u[k-1][ic][jc]
    
    #first inject to points coinciding with coarse points
    for ic in range (1,iic): 
        for jc in range (1,jjc):
            u[k][2*ic][2*jc] = u[k-1][ic][jc]
            
    #interpolate intermediate y direction
    for i in range (2,ii[k]-1): 
        u[k][i][1]    = (5.0*u[k][i][0  ]+15.0*u[k][i][2   ]-5.0*u[k][i][4   ]+u[k][i][6   ])*.0625
        for j in range (3,jj[k]-2,2): 
            u[k][i][j]   = (-u[k][i][j-3]+9.0 *u[k][i][j-1 ]+9.0*u[k][i][j+1 ]-u[k][i][j+3 ])*.0625        
        u[k][i][jj[k]-1] = (5.0*u[k][i][jj[k]]+15.0*u[k][i][jj[k]-2]-5.0*u[k][i][jj[k]-4]+u[k][i][jj[k]-6])*.0625

    #interpolate in x direction 
    for j in range (1,jj[k]): 
        u[k][1][j]    = (5.0*u[k][0  ][j]+15.0*u[k][2   ][j]-5.0*u[k][4   ][j]+u[k][6   ][j])*.0625
        for i in range (3,ii[k]-2,2): 
            u[k][i][j]   = (-u[k][i-3][j]+9.0 *u[k][i-1 ][j]+9.0*u[k][i+1 ][j]-u[k][i+3 ][j])*.0625
        u[k][ii[k]-1][j] = (5.0*u[k][ii[k]][j]+15.0*u[k][ii[k]-2][j]-5.0*u[k][ii[k]-4][j]+u[k][ii[k]-6][j])*.0625    
    return u, uconv

def conver(u, uconv, k, maxlevel):
    '''
    convergence check using converged solution on level k 
    and on next coarser grid k-1
    '''
    
    iic = ii[k-1]
    jjc = jj[k-1]
    
    if (k==(maxlevel-1)):
        err = 0.0 
        for ic in range (1,iic): 
            for jc in range (1,jjc):
                err += np.fabs(uconv[k-1][ic][jc]-u[k][2*ic][2*jc])
    else:
        err = 0.0 
        for ic in range (1,iic): 
            for jc in range (1,jjc):
                err += np.fabs(uconv[k-1][ic][jc]-uconv[k][2*ic][2*jc])
    
    return (err/((iic-1)*(jjc-1)))


# In[ ]:

'''
MULTIGRID DRIVING ROUTINES
'''
def cycle(u, uold, f, k, wu, nu0, nu1, nu2, gamma, EPSILON):
    '''
    perform coarse grid correction cycle starting on level k  
    nu1 pre-relaxations, nu2 post relaxation, nu0 relaxations 
    on the coarest grid, cycleindex gamma=1 for Vcycle,       
    gamma=2 for Wcycle
    '''  
    if (k==0):#base case
        for i in range(nu0):
            [u,wu] = relax(u, f, k, ii, jj, hx, hy, wu, EPSILON)
    else:
        for i in range(nu1):        
            [u,wu]  = relax(u, f, k, ii, jj, hx, hy, wu, EPSILON)
        [u,uold]= coarsen_u(u, uold, k, ii, jj, hx, hy, wu, EPSILON)
        f = coarsen_f(u, f, k, ii, jj, hx, hy, wu, EPSILON)
        for i in range(gamma): 
            [u,wu] = cycle(u, uold, f, k-1, wu, nu0, nu1, nu2, gamma, EPSILON)
        u = refine(u, uold, f, k, ii, jj, hx, hy, EPSILON)
        for i in range(nu2):
            [u,wu]  = relax(u, f, k, ii, jj, hx, hy, wu, EPSILON)
    
    return u,wu

def fmg (u, uold, uconv, f, k, wu, maxlevel, nu0, nu1, nu2, gamma, ncy, EPSILON):
    '''
    perform FMG with k levels and ncy cycles per level

    '''  
    
    if (maxlevel==1):#base case
        for j in range(ncy):
            for i in range(nu0):
                [u,wu] = relax(u, f, k, ii, jj, hx, hy, wu, EPSILON)
    else:
        if (k==0):#base case
            for i in range(nu0):
                [u,wu] = relax(u, f, k, ii, jj, hx, hy, wu, EPSILON)
        else:
            [u, uconv, wu] = fmg (u, uold, uconv, f, k-1, wu, maxlevel, nu0, nu1, nu2, gamma, ncy, EPSILON)
            [u,uconv] = fmg_interpolate(u, uconv, f, k, ii, jj, hx, hy, EPSILON)
            for j in range(ncy):
                [u,wu] = cycle(u, uold, f, k, wu, nu0, nu1, nu2, gamma, EPSILON)
                print("\n")
    return u, uconv, wu


# In[ ]:

'''
MAIN PROGRAM
'''
#Stack
[nx0,ny0,xa,xb,ya,yb,wu] = [4,4,0.0,1.0,0.0,1.0,0.0]
#cycle parameters
[nu0,nu1,nu2,gamma]      = [10,2,1,1]
EPSILON  = 1
maxlevel = input('\ngive maxlevel: ')
ncy      = input('\ngive ncy: ') 
[ii,jj,hx,hy,f,u,uold,uconv] = initialize(nx0, ny0, maxlevel, xa, xb, ya, yb)
for k in range(maxlevel):
    [u,f] = init_uf(u, f, k, ii, jj, xa, ya, hx, hy, EPSILON) 
[u, uconv, wu] = fmg (u, uold, uconv, f, k, wu, maxlevel, nu0, nu1, nu2, gamma, ncy, EPSILON)
print("\n\nLevel %d: er=%10.8e\n\n"%(maxlevel,conver_a(u, maxlevel-1, ii, jj, xa, ya, hx, hy)))
if (maxlevel>1):
    print("\n")
    for j in range(1,maxlevel):
        print("\n aen(%2d,%2d)=%8.5e"%(j+1,j,conver(u, uconv, j, maxlevel)))
print("\n")

