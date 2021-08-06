import numpy as np
import random
import math
from scipy.special import beta, betainc
from scipy.integrate import quad as integrate
from scipy.integrate import dblquad
from numpy import sqrt
from math import asin
from numba import jit


# beta distribution function 
@jit(nopython=True)
def BetaPDF(t,a,b):
    if t<0 or t>1:
        return 0.
    else:
        return 1/beta(a,b)*t**(a-1)*(1-t)**(b-1)

#
#
# Quantile and mean quantile functions
#
#    

@jit(nopython=True)
def QUnif(t, c, r):
    assert (t>0 and t<1), "t must be in [0,1]"
    return (c-r + 2*t*r)

@jit(nopython=True)
def QTri(t, c, r, m):
    assert (t>0 and t<1), "t must be in [0,1]"
    assert (m>c-r and m<c+r), "m must be in [c-r,c+r]"
    thresh = (m-c+r)/(2*r)
    if (t <= thresh):
        Q = c-r + sqrt(2*r*(m-c+r)*t)
    else:
        Q = c+r - sqrt(2*r*(c+r-m)*(1-t))
    return Q

# Mean quantile function
@jit(nopython=True)
def mQTri(t,c,r,a,b):
    assert (a>=0 and b>=0), "a and b must be positive"
    if (a==1 and b==1):
        part1 = 2*r*t+c-r
        part2 = 4/3.*r*(sqrt(t) - sqrt(1-t) - t**2 + (1-t)**2)
    else:
        part1 = 2*r * BetaPDF(t,a,b) + c - r
        part2 = 2*r / beta(a,b) * ( sqrt(t) * ( beta(a+1/2,b) - betainc(a+1/2,b,t)  ) - sqrt(1-t) * betainc(a,b+1/2,t) )
    return part1 + part2    

#
#
# Mallows distances
#
#

# Assume the 1st interval is given by [c1-r1, c1+r1]
# Assume the 2nd interval is given by [c2-r2, c2+r2]


##
## Assume the data are uniformly distributed over the 1st interval
## Assume the data are uniformly distributed over the 2nd interval
##
@jit(nopython=True)
def MalUnifUnif(c1,r1,c2,r2):
    return sqrt((c1 - c2)**2 + 1./3*(r1-r2)**2)

##
## Assume the data are uniformly distributed over the 1st interval
## Assume the data are triangular distributed over the 2nd interval
##
@jit(nopython=True)
def MalUnifTri(c1,r1,c2,r2,m2):
    assert (m2>(c2-r2) and m2<(c2+r2)), "Mode m2 must be within [c2-r2,c2+r2]"
    val1 = c1-c2-r1+r2
    val2 = c1-c2-r1-r2
    pt1 = 4/3.*r1**2 + 2 * val2*r1 + val1**2 / (2*r2) * (m2-c2+r2)
    pt2 = val2**2 / (2*r2) * (c2-m2+r2)
    pt3 = (m2-c2+r2)**2 * (r1 - 2/3.*val1)/r2
    pt4 = (c2-m2+r2)**2 * (4/3.*r1 + 2/3.*val2)/r2
    pt5 = (1/(4*r2) -2*r1/(5*r2**2)) * ( (m2-c2+r2)**3 + (c2-m2+r2)**3 )
    return sqrt(pt1 + pt2 + pt3 + pt4 + pt5)

##
## Assume the data are triangular distributed over the 1st interval
## Assume the data are triangular distributed over the 2nd interval
##
@jit(nopython=True)
def MalTriTri(c1,r1,m1,c2,r2,m2):
    assert (m1>(c1-r1) and m1<(c1+r1)), "Mode m1 must be within [c1-r1,c1+r1]"
    assert (m2>(c2-r2) and m2<(c2+r2)), "Mode m2 must be within [c2-r2,c2+r2]"
    Index = ( (m1-c1)/r1 <= (m2-c2)/r2 )
  
    part0 = (c1-c2)**2 + 1/6.*((r1-r2)**2 + (m1-c1)**2 + (m2-c2)**2) - 5/3.*r1*r2
    if (Index):
        pt1A = c1-c2+r2
        pt1B = c1-c2+r1
        pt2 = (5 - (m1-c1)/r1)
        pt3 = (5 + (m2-c2)/r2)
        part4 = sqrt(r1*r2*(c1-m1+r1)*(m2-c2+r2))/2*(asin((m2-c2)/r2) - asin((m1-c1)/r1))
    else:
        pt1A = c1-c2-r2
        pt1B = c1-c2-r1    
        pt2 = (5 - (m2-c2)/r2)
        pt3 = (5 + (m1-c1)/r1)
        part4 = sqrt(r1*r2*(c2-m2+r2)*(m1-c1+r1))/2*(asin((m1-c1)/r1) - asin((m2-c2)/r2))
        
    part1 = 2*(m1-c1)*pt1A/3 - 2*(m2-c2)*pt1B/3
    part2 = sqrt(r1*r2*(m1-c1+r1)*(m2-c2+r2))/6 * pt2
    part3 = sqrt(r1*r2*(c1-m1+r1)*(c2-m2+r2))/6 * pt3
    return sqrt(part0 + part1 + part2 + part3 + part4)

#
#
# Monte Carlo estimates instead on integration
#
#
# beta random variate function
@jit
def BetaRVS(a,b,size):
    t = np.linspace(0, 1, 1001)
    p = [BetaPDF(i, a, b) for i in t]
    sump = np.sum(p)
    p = [p/sump for p in p]
    return np.random.choice(t, p=p, size=size) 

# Sample m1 and m2 from quantiles of beta distribution and calculate mean over these values
@jit
def MCMalUnifTri(c1,r1,c2,r2,a2,b2):
    m2 = BetaRVS(a2, b2, size=101)
    d = [MalUnifTri(c1,r1,c2,r2,m2[i]) for i in range(len(m2))]
    return np.mean(d)

@jit
def MCMalTriTri(c1,r1,a1,b1,c2,r2,a2,b2):
    m1 = BetaRVS(a1, b1, size=101)
    m2 = BetaRVS(a2, b2, size=101)
    d = [MalTriTri(c1,r1,i,c2,r2,j) for i in m1 for j in m2]
    return np.mean(d)


#
#
# Mean Mallows distances
#
#

# Assume the 1st interval is given by [c1-r1, c1+r1]
# Assume the 2nd interval is given by [c2-r2, c2+r2]
#
# Method 1: compute the Mallows distance using mean quantile functions
# Method 2: compute the mean of the Mallows distances

##
## Assume the data are uniformly distributed over the 1st interval
## Assume the data are triangular distributed over the 2nd interval
##
## We consider (m2-c2+r2)/(2*r2) to be beta distributed with parameters a2 and b2
## If a2=b2=1 then the mode is uniformly distributed on [c2-r2, c2+r2]

def M2MalUnifTriIntegrand(c1, r1, c2, r2, a2, b2):
    return lambda m2: MalUnifTri(c1, r1, c2, r2, m2)**2 * BetaPDF((m2-c2+r2)/(2*r2), a2, b2) / (2*r2)

def M1MalUnifTriIntegrand(c1, r1, c2, r2, a2, b2):
    return lambda t: (QUnif(t, c1, r1) - mQTri(t, c2, r2, a2, b2) )**2


def M1MalUnifTri(c1,r1,c2,r2,a2,b2):
    assert (a2>=0 and b2>=0), "a and b must be positive"
    
    if (a2==1 and b2==1):
        return sqrt((c1-c2)**2 - 22/45.*r1*r2 +1/3.*r1**2 - (20*math.pi -71)/45.*r2**2)
    else:
        return sqrt(integrate(M1MalUnifTriIntegrand(c1, r1, c2, r2, a2, b2), 0, 1)[0])

def M2MalUnifTri(c1,r1,c2,r2,a2,b2):
    assert (a2>=0 and b2>=0), "a and b must be positive"
    
    if (a2==1 and b2==1):
        return sqrt((c1-c2)**2 - 22*r1*r2/45 +r1**2/3 + 2*r2**2/9)
    else:
        return sqrt(integrate(M2MalUnifTriIntegrand(c1, r1, c2, r2, a2, b2), c2-r2, c2+r2)[0])
    
##
## Assume the data are triangular distributed over the 1st interval
## Assume the data are triangular distributed over the 2nd interval
##
## We consider (m1-c1+r1)/(2*r1) to be beta distributed with parameters a1 and b1
## If a1=b1=1 then the mode is uniformly distributed on [c1-r1, c1+r1]
## We consider (m2-c2+r2)/(2*r2) to be beta distributed with parameters a2 and b2
## If a2=b2=1 then the mode is uniformly distributed on [c2-r2, c2+r2]

def M1MalTriTriIntegrand(c1,r1,a1,b1,c2,r2,a2,b2):
    return lambda t: (mQTri(t, c1, r1, a1, b1) - mQTri(t, c2, r2, a2, b2) )**2

def M1MalTriTri(c1,r1,a1,b1,c2,r2,a2,b2):
    assert (a1>=0 and b1>=0), "a and b must be positive"
    assert (a2>=0 and b2>=0), "a and b must be positive"
    if (a1==1 and b1==1 and a2==1 and b2==1):
        return sqrt( (c1-c2)**2 - (20*math.pi-71)/45.*(r2-r1)**2 )
    else:
        return sqrt(integrate(M1MalTriTriIntegrand(c1,r1,a1,b1,c2,r2,a2,b2), 0, 1)[0])
  
def M2MalTriTriDblIntegrand(c1,r1,a1,b1,c2,r2,a2,b2):
    return lambda m1, m2: MalTriTri(c1, r1, m1, c2, r2, m2)**2 * BetaPDF((m1-c1+r1)/(2*r1), a1, b1) / (2*r1) * BetaPDF((m2-c2+r2)/(2*r2), a2, b2) / (2*r2)

def M2MalTriTri(c1,r1,a1,b1,c2,r2,a2,b2):
    assert (a1>=0 and b1>=0), "a and b must be positive"
    assert (a2>=0 and b2>=0), "a and b must be positive"
    if (a1==1 and b1==1 and a2==1 and b2==1):
        return sqrt( (c1-c2)**2 + 2*(r1**2+r2**2)/9 + (8*math.pi/9 - 142/45)*r1*r2 )
    else:
        return sqrt( dblquad(M2MalTriTriDblIntegrand(c1,r1,a1,b1,c2,r2,a2,b2) , c2-r2, c2+r2, c1-r1, c1+r1 )[0] )
    

#
#
# Mallows distance matrix for clustering given two sets of symbolic data (centres and ranges)
#
#
def MallowsDistMatrix(c1, r1, c2, r2, scenario="XunifYunif", method=1, a1=1., b1=1., a2=1., b2=1.):
    """
    ci: the centres of the intervals. A concatenation of ci centres of X intervals and ci_dash intervals of Y intervals.
    ri: ranges of the intervals. A concatenation of ri ranges of X intervals and ri_dash ranges of Y intervals.
    scenario: One of "XunifYunif": both X and Y intervals are uniformly distributed.
               "XunifYsym" : X uniform, Y symmetric.
               "XsymYsym"  : both X and Y intervals are symmetric.
               "Xunifskew" : X uniform, Y skewed (mode follow skewed beta based on a2, b2)
               "Xskewskew" : X skewed (mode follows skewed beta based on a1, b1), Y skewed (mode follow skewed beta based on a2, b2)
    method: 1 or 2. Only applicable to scenarios containing symmetic/skewed distributions (estimated using triangular distributions with modes m~beta(a,b))
                1: Calculate single mallows distance based on mean triangular distribution quantile function
                2: Mean of mallows distances based on varying triangular quantile function
    a1, b1, a2, b2: float parameters of beta distributions
    """
    allowed_scenarios = ['XunifYunif', 'XunifYsym', 'XsymYsym', 'XunifYskew', 'XsymYskew', 'XskewYskew']
    assert scenario in allowed_scenarios, "scenario must be one of 'XunifYunif', 'XunifYsym', or 'XsymYsym'"
    assert (method==1 or method==2), "method must be 1 or 2"
    a1, a2, b1, b2 = a1/1.,a2/1.,b1/1.,b2/1.
    size1 = len(c1)
    size2 = len(c2)
    size = size1 + size2
    D_M = np.zeros((size,size))
    # Calculate the D_M in four quadrants
    D_M_1 = np.zeros((size1,size1))
    D_M_2 = np.zeros((size1,size2))
    # D_M_3 is the same as D_M_2
    D_M_4 = np.zeros((size2,size2))
    # first quadrant
    for i in range(size1):
        for j in range(i+1,size1):
            if (scenario in ['XunifYunif', 'XunifYsym', 'XunifYskew']):
                D_M_1[i,j] = MalUnifUnif(c1[i],r1[i],c1[j],r1[j])
            elif (scenario in ['XsymYsym', 'XsymYskew']):
                if (method==1):
                    D_M_1[i,j] = M1MalTriTri(c1[i],r1[i],1.,1.,c1[j],r1[j],1.,1.)
                else:
                    D_M_1[i,j] = M2MalTriTri(c1[i],r1[i],1.,1.,c1[j],r1[j],1.,1.)
            elif (scenario in ['XskewYskew']):
                if (method==1):
                    D_M_1[i,j] = M1MalTriTri(c1[i],r1[i],a1,b1,c1[j],r1[j],a1,b1)
                else:
                    D_M_1[i,j] = M2MalTriTri(c1[i],r1[i],a1,b1,c1[j],r1[j],a1,b1)
    D_M_1 = D_M_1 + D_M_1.T # no need to subtract diagonal (np.diag(np.diag(D_M))) since they are zero anyway
    # second (and third) quadrant
    for i in range(size1):
        for j in range(size2):
            if (scenario=='XunifYunif'):
                D_M_2[i,j] = MalUnifUnif(c1[i],r1[i],c2[j],r2[j])
            elif (scenario=='XunifYsym'):
                if (method==1):
                    D_M_2[i,j] = M1MalUnifTri(c1[i],r1[i],c2[j],r2[j],1.,1.) # beta(1,1) is uniform
                else:
                    D_M_2[i,j] = M2MalUnifTri(c1[i],r1[i],c2[j],r2[j],1.,1.) # beta(1,1) is uniform
            elif (scenario=='XsymYsym'):
                if (method==1):
                    D_M_2[i,j] = M1MalTriTri(c1[i],r1[i],1.,1.,c2[j],r2[j],1.,1.)
                else:
                    D_M_2[i,j] = M2MalTriTri(c1[i],r1[i],1.,1.,c2[j],r2[j],1.,1.)
            elif (scenario=='XunifYskew'):
                if (method==1):
                    D_M_2[i,j] = M1MalUnifTri(c1[i],r1[i],c2[j],r2[j],a2,b2)
                else:
                    D_M_2[i,j] = M2MalUnifTri(c1[i],r1[i],c2[j],r2[j],a2,b2)# M2MalUnifTri(c1[i],r1[i],c2[j],r2[j],a2,b2)
            elif (scenario=='XsymYskew'):
                if (method==1):
                    D_M_2[i,j] = M1MalTriTri(c1[i],r1[i],1.,1.,c2[j],r2[j],a2,b2)
                else:
                    D_M_2[i,j] = M2MalTriTri(c1[i],r1[i],1.,1.,c2[j],r2[j],a2,b2)
            elif (scenario=='XskewYskew'):
                if (method==1):
                    D_M_2[i,j] = M1MalTriTri(c1[i],r1[i],a1,b1,c2[j],r2[j],a2,b2)
                else:
                    D_M_2[i,j] = M2MalTriTri(c1[i],r1[i],a1,b1,c2[j],r2[j],a2,b2)
    D_M_3 = D_M_2.T
    # fourth quadrant
    for i in range(size2):
        for j in range(i+1,size2):
            if (scenario in ['XunifYunif']):
                D_M_4[i,j] = MalUnifUnif(c2[i],r2[i],c2[j],r2[j])
            elif (scenario in ['XunifYsym', 'XsymYsym']):
                if (method==1):
                    D_M_4[i,j] = M1MalTriTri(c2[i],r2[i],1.,1.,c2[j],r2[j],1.,1.)
                else:
                    D_M_4[i,j] = M2MalTriTri(c2[i],r2[i],1.,1.,c2[j],r2[j],1.,1.)
            elif (scenario in ['XunifYskew', 'XsymYskew', 'XskewYskew']):
                if (method==1):
                    D_M_4[i,j] = M1MalTriTri(c2[i],r2[i],a1,b1,c2[j],r2[j],a2,b2)
                else:
                    D_M_4[i,j] = M2MalTriTri(c2[i],r2[i],a1,b1,c2[j],r2[j],a2,b2)
    D_M_4 = D_M_4 + D_M_4.T # no need to subtract diagonal (np.diag(np.diag(D_M))) since they are zero anyway
    
    D_M[:size1,:size1] = D_M_1
    D_M[:size1,size1:] = D_M_2
    D_M[size1:,:size1] = D_M_3
    D_M[size1:,size1:] = D_M_4
    
    return D_M 