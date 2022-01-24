import numpy as np
import math
from math import *
import sympy
import numba
from numba import jit

def convertEllipseToImplicitForm(a,b,x0,y0,phi):
    A = (a**2)*sin(phi)**2+b**2*cos(phi)**2
    B = 2*(b**2-a**2)*sin(phi)*cos(phi)
    C = (a**2)*cos(phi)**2+(b**2)*sin(phi)**2
    D = -2*A*x0-B*y0
    E = -B*x0-2*C*y0
    F = A*x0**2+B*x0*y0+C*y0**2-(a*b)**2
    
    B=B/A
    C=C/A
    D=D/A
    E=E/A
    F=F/A
    A=1

    equation = np.array([A,B,C,D,E,F])
    return equation

@jit(nopython=True)
def convertEllipseToGeometricForm(equation):


    a=equation[0]
    b=equation[1]
    c=equation[2]
    d=equation[3]
    e=equation[4]
    f=equation[5]

     
    delta = b**2 - 4*a*c
    lambdaPlus = 0.5*(a + c - (b**2 + (a - c)**2)**0.5)
    lambdaMinus = 0.5*(a + c + (b**2 + (a - c)**2)**0.5)

    psi = b*d*e - a*e**2 - b**2*f + c*(4*a*f - d**2)
    Vplus = (psi/(lambdaPlus*delta))**0.5
    Vminus = (psi/(lambdaMinus*delta))**0.5

    # major semi-axis
    axisA = max(Vplus,Vminus)
    # minor semi-axis
    axisB = min(Vplus,Vminus)

    # determine coordinates of ellipse centroid
    xCenter = (2*c*d - b*e)/(delta)
    yCenter = (2*a*e - b*d)/(delta)

    # angle between x-axis and major axis 
    tau = 0
    # determine tilt of ellipse in radians
    if (Vplus >= Vminus):
      if(b == 0 and a < c):
          tau = 0
      elif (b == 0 and a >= c):
          tau = 0.5*pi
      elif (b < 0 and a < c):
          tau = 0.5*np.arctan(b/(a - c))
      elif (b < 0 and a == c):
          tau = pi/4
      elif (b < 0 and a > c):
          tau = 0.5*np.arctan(b/(a - c))+ pi/2
      elif (b > 0 and a < c):
          tau = 0.5*np.arctan(b/(a - c))+ pi
      elif (b > 0 and a == c):
          tau = pi*(3/4);
      elif (b > 0 and a > c):
          tau = 0.5*np.arctan(b/(a - c))+ pi/2
    elif (Vplus < Vminus):
      if(b == 0 and a < c):
           tau = pi/2
      elif (b == 0 and a >= c):
               tau = 0
      elif (b < 0 and a < c):
          tau = 0.5*np.arctan(b/(a - c))+ pi/2
      elif (b < 0 and a == c):
           tau = pi*(3/4)
      elif (b < 0 and a > c):
          tau = 0.5*np.arctan(b/(a - c))+ pi
      elif (b > 0 and a < c):
          tau = 0.5*np.arctan(b/(a - c))+ pi/2
      elif (b > 0 and a == c):
          tau = pi/4
      elif (b > 0 and a > c):
          tau = 0.5*np.arctan(b/(a - c))

    return axisA, axisB, xCenter, yCenter, tau
    
# routine that finds both the distance from a point to an ellipse, and the point on the ellipse closest to the provided point.
@jit(nopython=True)
def distanceToEllipse(p,equation):

    # we have to use a dual approach here - one approach breaks down
    # as p approachs the center of the ellipse - so we have to detect
    # that situation and deal with it separately

    axisA, axisB, xCenter, yCenter, tau = convertEllipseToGeometricForm(equation)
    if np.linalg.norm(p-np.array([xCenter,yCenter]))<0.01*axisB:
        # in this case we are close enough to the ellipse center
        # for the other approach to have issues.  At the same time,
        # we are close enough that the answer should be about the same
        # as if we really WERE in the center - in which case we know
        # the distance is axisB and the point is
        # x_center +/- b e_tau_T

        e_tau_T = np.array([-sin(tau),cos(tau)])
        p_close = np.array([xCenter,yCenter])+e_tau_T*axisB
        dist = axisB

    else:
    

        A=equation[0]
        B=equation[1]
        C=equation[2]
        D=equation[3]
        E=equation[4]
        F=equation[5]

        x0=p[0]
        y0=p[1]

        a = A*x0**2 + B*x0*y0 + D*x0 + C*y0**2 + E*y0 + F
        b = D*(D + B*y0 - 2*C*x0) - F*(4*A + 4*C) + E*(E - 2*A*y0 + B*x0) - D*x0*(2*A + 2*C) - E*y0*(2*A + 2*C) + 2*A*x0*(D + B*y0 - 2*C*x0) + B*x0*(E - 2*A*y0 + B*x0) + B*y0*(D + B*y0 - 2*C*x0) + 2*C*y0*(E - 2*A*y0 + B*x0)
        c = A*(2*x0*(B*E - 2*C*D) + (D + B*y0 - 2*C*x0)**2) - C*(2*y0*(2*A*E - B*D) - (E - 2*A*y0 + B*x0)**2) - E*(2*A*E - B*D) + D*(B*E - 2*C*D) + F*((2*A + 2*C)**2 - 2*B**2 + 8*A*C) - D*x0*(B**2 - 4*A*C) - E*y0*(B**2 - 4*A*C) - D*(2*A + 2*C)*(D + B*y0 - 2*C*x0) - E*(2*A + 2*C)*(E - 2*A*y0 + B*x0) - B*x0*(2*A*E - B*D) + B*y0*(B*E - 2*C*D) + B*(E - 2*A*y0 + B*x0)*(D + B*y0 - 2*C*x0)
        d = 2*F*(2*A + 2*C)*(B**2 - 4*A*C) - D*(B**2 - 4*A*C)*(D + B*y0 - 2*C*x0) - E*(B**2 - 4*A*C)*(E - 2*A*y0 + B*x0) + E*(2*A*E - B*D)*(2*A + 2*C) - D*(B*E - 2*C*D)*(2*A + 2*C) - B*(2*A*E - B*D)*(D + B*y0 - 2*C*x0) - 2*C*(2*A*E - B*D)*(E - 2*A*y0 + B*x0) + 2*A*(B*E - 2*C*D)*(D + B*y0 - 2*C*x0) + B*(B*E - 2*C*D)*(E - 2*A*y0 + B*x0)
        e = A*(B*E - 2*C*D)**2 + C*(2*A*E - B*D)**2 + F*(B**2 - 4*A*C)**2 + E*(2*A*E - B*D)*(B**2 - 4*A*C) - D*(B*E - 2*C*D)*(B**2 - 4*A*C) - B*(2*A*E - B*D)*(B*E - 2*C*D)

        # numba has a tendancy to "freak out" whenever a function taking in an array of real numbers outputs
        # an array of complex numbers.  One example is the "roots" function which takes in the coefficients
        # of a polynomial and outputs the roots.  In general, a real polynomial will have complex roots.
        # But this will cause numba to crash.  So, in order to placate numba,
        # we have to artificially make the coefficients of our polynomial
        # complex by adding on 1j*0.

        poly_coeff = np.zeros(5,dtype=np.complex128)
        
        poly_coeff = np.array([a, b, c, d, e])+1j*np.zeros(5)
        r = np.roots(poly_coeff)
        
        dist2 = np.float64(inf)
        
        x_close=-1
        y_close=-1
        
        for k in range(5):
        
            lamba = r[k]

            # however, we actually are only interested in (non-zero) real roots - so we look for roots with a vanishingly small imaginary part.           
            if abs(lamba)!=0 and abs(lamba.imag)<1e-10:
            
                Det = B**2-(2*A-lamba)*(2*C-lamba)

                if Det!=0:

                    x = (2*C-lamba)*(lamba*x0+D)-B*(lamba*y0+E)
                    y = (2*A-lamba)*(lamba*y0+E)-B*(lamba*x0+D)

                    x = x/Det
                    y = y/Det


                    
                    distCand2 = (x-x0)**2+(y-y0)**2;
                    # distCand2 should be real and positive.  However, due to rounding error it may have a very small imaginary part, and more importantly numba believes it is complex.
                    # So we have have to talk about the "real part" of distCand2.
                    
                    if distCand2.real < dist2 and 0<=distCand2.real and abs(distCand2.imag)<1e-10:
                        dist2 = distCand2.real
                        # again, x and y should be real already - but numba believes them to be complex, because they were computed in terms of lambda.  So we have to take real parts to avoid
                        # a type error.
                        x_close = x.real
                        y_close = y.real
        p_close = np.array([x_close,y_close],dtype=np.float64)
        dist = abs(dist2**0.5)
    return p_close, dist

# routine to get the leftmost, rightmost, topmost, bottommost points on an ellipse.
def getEllipseExtremePoints(equation):

    A=equation[0]
    B=equation[1]
    C=equation[2]
    D=equation[3]
    E=equation[4]
    F=equation[5]

    a = C-(B**2)/(4*A)
    b = E-D*B/(2*A)
    c = F-(D**2)/(4*A)
    
    y1 = (-b+sqrt(b**2-4*a*c))/(2*a)
    
    y2 = (-b-sqrt(b**2-4*a*c))/(2*a)
    
    if y1>y2:
        y_top = y1
        y_bottom = y2
    else:
        y_top = y2
        y_bottom = y1
    
    p_top = np.array([-(B*y_top+D)/(2*A),y_top])
    p_bottom = np.array([-(B*y_bottom+D)/(2*A),y_bottom])
    
    a = A-(B**2)/(4*C)
    b = D-E*B/(2*C)
    c = F-(E**2)/(4*C)
    
    x1 = (-b+sqrt(b**2-4*a*c))/(2*a)
    
    x2 = (-b-sqrt(b**2-4*a*c))/(2*a)
    
    if x1>x2:
        x_right = x1
        x_left = x2
    else:
        x_right = x2
        x_left = x1
    
    p_right = np.array([x_right,-(B*x_right+E)/(2*C)])
    p_left = np.array([x_left,-(B*x_left+E)/(2*C)])
    return p_top,p_bottom,p_left,p_right

def not_degenerate_ellipse(equation):

    # if the minor axis is less that tol pixels, we're calling the ellipse
    # degenerate.
    
    tol = 3 # set the tol to 3 pixels for now
    
    a,b,_,_,_=convertEllipseToGeometricForm(equation)
    
    
    if min(a,b)<tol:
        return False
    else:
        return True

# routine to find the intersection of an ellipse with a line.  To be more precise, it is assumed that we have not an ellipse
# but rather an elliptic arc.  It is further assumed that the line and elliptic arc are oriented such that the intersection point is
# unique - i.e. we have something like this:

#   |   /    |
#    \_/____/
#     /
#    /

# and not this

# ___|_____|_____
#    \_____/
#     
# But, I am not doing any explicit checks for this - rather I am using imperfect heuristics.
# Therefore, the routine will return a single intersection point even if in
# reality there should be two.  But, for our purposes, this should never come up.

def get_intersection_with_line(equation,p,v,concavity):
    # I'm assuming the line is of the form l(t) = p+t*v
    
    # I'm assuming we have a well defined concavity so that the
    # intersection is unique.

    vx=v[0]
    vy=v[1]
    px=p[0]
    py=p[1]

    A = equation[0]; B = equation[1]; C = equation[2]; D = equation[3]; E = equation[4]; F = equation[5]

    # a line intersects an ellipse at at most two points, according to the
    # following quadratic.
    
    a=A*vx**2 + B*vx*vy + C*vy**2;
    b=D*vx + E*vy + 2*A*px*vx + B*px*vy + B*py*vx + 2*C*py*vy
    c=A*px**2 + B*px*py + D*px + C*py**2 + E*py + F

    desc = b**2-4*a*c

    if desc < 0: # we don't intersect the ellipse
        p_int = None;
        int_found = False;
    else:
        
        t1 = (-b+sqrt(desc))/(2*a)
        t2 = (-b-sqrt(desc))/(2*a)
        
        p1 = p+t1*v
        p2 = p+t2*v

        # we want a unique intersection point, so we use the concavity of the ellipse
        # to disambiguate.  
        
        if concavity=='up': # only the lower half of the ellipse is visible
            # the intersection with the smaller y-coordinate should be the only true one.
            if p1[1] < p2[1]:
                p_int = p1
            else:
                p_int = p2
            int_found = True
        
        elif concavity=='down': # only the upper half of the ellipse is visible
            # the intersection with the larger y-coordinate should be the only true one.
            if p1[1] > p2[1]:
                p_int = p1
            else:
                p_int = p2
            int_found = True
            
        elif concavity=='leftwards': # only the right half of the ellipse is visible
            # the intersection with the larger x-coordinate should be the only true one.
            if p1[0]> p2[0]:
                p_int = p1
            else:
                p_int = p2
            int_found = True
            
        elif concavity=='rightwards': # only the left half of the ellipse is visible
            # the intersetion with the smaller x-coordinate should be the only true one.
            if p1[0]< p2[0]:
                p_int = p1
            else:
                p_int = p2
            int_found = True
        else:
            p = None
            int_found = False
    return p_int,int_found

