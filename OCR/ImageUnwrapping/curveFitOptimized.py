import numpy as np
import scipy.linalg
import math
from dataclasses import dataclass
import OCR.ImageUnwrapping.Ellipse as Ellipse
import numba
from numba import jit

# this is a simple implementation of the "direct ellipse fit" method
# described in the paper "Direct Least Squares Fitting of Ellipses"
# by Andrew W. Fitzgibbon, Maurizio Pilu, Robert B. Fisher
# which may be read online for free at
# http://cseweb.ucsd.edu/~mdailey/Face-Coord/ellipse-specific-fitting.pdf

# This method involves solving a generalized eigenvalue problem of the form S*x = lambda*C*x.
# However, in order to accelerate the method using numbda which doesn't support generalized eigenvalue
# problems (it supports numpy.eig, but not scipy.eig - the latter does generalized eigenvalue problems while the
# former does not), I had to do some fiddling to turn the generalized eigenvalue problem into a regular one.
@jit(nopython=True)
def direct_ellipse_fit(points):
    # I'm assuming "points" is a N x 2 numpy array.

    (rows,cols)=points.shape;
    if not cols ==2:
        raise ValueError('expected a N x 2 array')

    x = points[:,0]
    y = points[:,1]

    t = np.array([1, 0, 0, 0, 0, 1],dtype=np.float64)

    D = np.zeros((rows,6),dtype=np.float64)
    # these operations are all elementwise
    D[:,0]=x**2
    D[:,1]=x*y
    D[:,2]=y**2
    D[:,3]=x
    D[:,4]=y
    D[:,5]=np.ones(rows)

    S = (D.transpose()).dot(D)

    C = np.zeros((6,6),dtype=np.float64)
    C[0,2]=2
    C[2,0]=2
    C[1,1]=-1

    S_11 = S[0:3,0:3]
    S_12 = S[0:3,3:6]
    S_21 = S[3:6,0:3]
    S_22 = S[3:6,3:6]
    C_11 = C[0:3,0:3]
    C_11_inv = np.zeros((3,3),dtype=np.float64)
    C_11_inv[0,2]=0.5
    C_11_inv[2,0]=0.5
    C_11_inv[1,1]=-1

    # A is actually a real matrix.  However, we will later
    # pass it into numpy's "eig" function, which will in
    # general produce a complex result.  Python doesn't care
    # about this, but numba will complain about a "domain change"
    # (that is, from real to complex).  To get arround this,
    # we simply define A as complex to begin with, but set the
    # imaginary part to 0.
    A = np.zeros((3,3),dtype=np.complex128)

    A=C_11_inv.dot(S_11-S_12.dot(np.linalg.inv(S_22).dot(S_21)))+1j*np.zeros((3,3))

    # e_val and e_vec will in general be complex.
    e_val,e_vec = np.linalg.eig(A)

    j_correct = -1

    # although e_val will in general be complex, the eigenvalue we care
    # about should be real, positive, and finite (moreover, there should only
    # be one eigenvalue satisfying both of these properties.
    for j in range(3):
        # checking the eigenvalue is "real", allowing for some roundoff error
        if abs(e_val[j].imag)<1e-7:
            # checking that the eigenvalue is positive and finite.
            if e_val[j].real>0 and e_val[j].real <math.inf:
                j_correct = j
    if j_correct == -1:
        raise ValueError('failed to find positive, finite eigenvalue');
    e_correct = np.zeros(3,dtype=np.float64)
    
    # the associated eigenvector should be real, so taking the real part
    # here should amount to discarding a non-existent imaginary part.
    e_correct = e_vec[:,j_correct].real
        
    mu =  e_correct.dot( C_11.dot(e_correct));
    mu = np.float64(1.0/mu**0.5)

    x_vec = e_correct
    y_vec = -np.linalg.inv(S_22).dot(S_21).dot(x_vec)
    

    
    theta =  mu*np.array([x_vec[0],x_vec[1],x_vec[2],y_vec[0],y_vec[1],y_vec[2]],dtype=np.float64)
    return theta

# It is well known that algebraic methods for ellipse fit - of which "direct ellipse fit"
# is a member, perform better if the coordinate system is first rescaled in the following way.

@jit(nopython=True)
def direct_ellipse_fit_using_normalization(points):
    # I'm assuming "points" is a N x 2 numpy array.
    (rows,cols)=points.shape
    if not cols ==2:
        raise ValueError('expected a N x 2 array')

    homogenous_points = np.zeros((rows,3))
    homogenous_points[:,0]=points[:,0]
    homogenous_points[:,1]=points[:,1]
    homogenous_points[:,2]=np.ones(rows)
    newpts,T=normalise2dpts(homogenous_points)

    theta = direct_ellipse_fit(newpts)
    theta = theta/np.linalg.norm(theta)
    a=theta[0];b=theta[1];c=theta[2]
    d=theta[3];e=theta[4];f=theta[5]

    C = np.array([[a,b/2,d/2],[b/2,c,e/2],[d/2,e/2,f]])

    # normalize C
    C = ((T.transpose()).dot(C)).dot(T)
    aa = C[0,0];bb=2*C[0,1];dd=2*C[0,2]
    cc = C[1,1];ee=2*C[1,2];ff=C[2,2]
    theta = np.array([aa,bb,cc,dd,ee,ff])
    theta = theta/np.linalg.norm(theta)
    return theta

@jit(nopython=True)    
def normalise2dpts(pts):
    # I'm assuming "points" is a N x 3 numpy array.
    # of "homogenous" points where the third component is 1.

    (rows,cols)=pts.shape
    if not cols ==3:
        raise ValueError('expected a N x 3 array')

    # manually do the average or jit will complain.
    cx = 0
    cy = 0
    for j in range(rows):
        cx=cx+pts[j,0]
        cy=cy+pts[j,1]
    cx = cx/rows
    cy = cy/rows
    
    new_pts = np.zeros((rows,2))
    new_pts[:,0]=pts[:,0]-cx
    new_pts[:,1]=pts[:,1]-cy
    dists = (new_pts[:,0]**2+new_pts[:,1]**2)**0.5 # elementwise

    # manually do average of jit will complain.
    avg_dist = 0
    for j in range(rows):
        avg_dist = avg_dist+dists[j]
    avg_dist = avg_dist/rows
    scale = math.sqrt(2)/avg_dist
    T = np.array([[scale, 0.0 , -scale*cx],[0.0, scale , -scale*cy],[0.0,0.0,1.0]],dtype=np.float64)
    normalized_points = T.dot(pts.transpose())
    normalized_points = normalized_points.transpose()
    points_out = np.zeros((rows,2))
    points_out[:,0]=normalized_points[:,0]
    points_out[:,1]=normalized_points[:,1]
    
    return points_out,T

# this (and the routines) below are an implementation of the ellipse fit routine described in the paper
# "Guaranteed Ellipse Fitting with the Sampson Distance" by Zygmunt L. Szpak, Wojciech Chojnacki, and Anton van den Hengel
# which can be viewed for free online at http://www.users.on.net/~zygmunt.szpak/pubs/ellipsefit-1.pdf.

# Actually, they provide a matlab implementation of their algorithm, which can be downloaded for free here:  http://www.users.on.net/~zygmunt.szpak/sourcecode.html

# All I have done is port their implementation from Matlab to python (also I corrected some small bugs).  However, their implementation uses a Matlab struct.  When I ported
# To python the first time, I replaced this with a Python dataclass.  However, in order to accelerate using numba, I had to remove the dataclass as numba doesn't support dataclasses.

@jit(nopython=True)
def compute_guaranteedellipse_estimates(points):
    # I'm assuming "points" is a N x 2 numpy array.
    (rows,cols)=points.shape;
    if not cols ==2:
        raise ValueError('expected a N x 2 array')

    homogenous_points = np.zeros((rows,3))
    homogenous_points[:,0]=points[:,0]
    homogenous_points[:,1]=points[:,1]
    homogenous_points[:,2]=np.ones(rows)
    newpts,T=normalise2dpts(homogenous_points)

    theta_in = direct_ellipse_fit(newpts)
    theta_in = theta_in/np.linalg.norm(theta_in)

    # I have to take the transpose of the points array because
    # guaranteeEllipseFit expects a 2 x N array of points, not
    # N x 2.

    transposed_points = newpts.transpose();
    #transposed_points = np.zeros((cols,rows),dtype=np.float64)
    #for j in range(rows):
    #    transposed_points[0,j]=newpts[j,0]
    #    transposed_points[1,j]=newpts[j,1]

    theta = np.zeros(6,dtype=np.float64)

    theta = guaranteedEllipseFit(theta_in,transposed_points)
    theta = theta/np.linalg.norm(theta)
    
    a=theta[0];b=theta[1];c=theta[2]
    d=theta[3];e=theta[4];f=theta[5]

    C = np.array([[a,b/2,d/2],[b/2,c,e/2],[d/2,e/2,f]])

    # normalize C
    C = ((T.transpose()).dot(C)).dot(T)
    aa = C[0,0];bb=2*C[0,1];dd=2*C[0,2]
    cc = C[1,1];ee=2*C[1,2];ff=C[2,2]
    theta = np.array([aa,bb,cc,dd,ee,ff])
    theta = theta/np.linalg.norm(theta)

    # now compute the error vector

    error = np.zeros(rows);
    for j in range(rows):
        p_close,dist = Ellipse.distanceToEllipse(points[j,:],theta)
        error[j]=dist
    
    return theta,error

@jit(nopython=True)    
def guaranteedEllipseFit(t_in,data_points):

    # data points should be 2 x N

    keep_going = True
    maxIter = 200

    _,dataCols = data_points.shape # the number of data points is equal to dataCols.  dataRows we don't actually care about.
    datapoints = data_points
    numberOfPoints=dataCols
    cost = np.zeros(maxIter,dtype=np.float64)
    t = np.zeros((6,maxIter),dtype=np.float64)
    delta = np.zeros((6,maxIter),dtype=np.float64)

    use_pseudoinverse= False
    theta_updated = False
    lamba = 0.01
    k = int(0)
    damping_multiplier= 1.2
    gamma = 0.00005
    F = np.array([[0,0,2,0,0,0],[0,-1,0,0,0,0],[2,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],dtype=np.float64)
    alpha = 1e-3
    I = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=np.float64)
    tolDelta = 1e-7
    tolCost = 1e-7
    tolTheta = 1e-7
    
    #struct = Struct(datapoints = data_points,numberOfPoints=dataCols,cost = np.zeros(maxIter),t = np.zeros((6,maxIter)),delta = np.zeros((6,maxIter)));
    t_in = t_in/np.linalg.norm(t_in)
    t[:,k]=t_in

    delta[:,k]=np.ones(6,dtype=np.float64)
    # main estimation loop
    while keep_going and k<maxIter-1:
        r = np.zeros((numberOfPoints+1),dtype=np.float64)
        jacobian_matrix = np.zeros((numberOfPoints,6),dtype=np.float64)
        jacobian_matrix_barrier = np.zeros(6,dtype=np.float64)
        jacobian_matrix_full = np.zeros((numberOfPoints+1,6),dtype=np.float64)
        t_current = t[:,k]
        for i in range(numberOfPoints):
            m = data_points[:,i]
            ux_j = np.array([m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1],dtype=np.float64)
            dux_j = np.array([[2*m[0],     m[1],     0,      1,    0,  0],[0,m[0],2*m[1],0,1,0]],dtype=np.float64)
            dux_j = dux_j.transpose()

            # outer product

            A = np.outer(ux_j,ux_j)

            B = dux_j.dot(dux_j.transpose())

            tBt = t_current.dot(B.dot(t_current))
            tAt = t_current.dot(A.dot(t_current))
            r[i]=(tAt/tBt)**0.5

            M = (A/tBt)
            Xbits = B*((tAt)/(tBt**2))
            X = M-Xbits
            # gradient for the AML cost function
            grad = (X.dot(t_current))/((tAt/tBt)**0.5)
            # build up the jacobian matrix
            jacobian_matrix[i,:]=grad
            jacobian_matrix_full[i,:]=grad

        # barrier term
        tIt = t_current.dot((I).dot(t_current))
        tFt = t_current.dot((F).dot(t_current))
        # add the penalty term
        r[numberOfPoints]=alpha*(tIt/tFt)

        # Derivative barrier component
        N = (I/tFt)
        Ybits = F *((tIt)/(tFt)**2)
        Y=N-Ybits
        grad_penalty = 2*alpha*(Y.dot(t_current))
        jacobian_matrix_barrier=grad_penalty
        # Jacobian Matrix after combining AML and barrier terms
        jacobian_matrix_full[numberOfPoints,:]=grad_penalty

        # approximate Hessian Matrix
        H = (jacobian_matrix_full.transpose()).dot(jacobian_matrix_full)
        # sum of squares cost for the current iteration
        
        cost[k] = (r).dot(r);

        #If we haven't overshot the barrier term then we use the LevenbergMarquadt step
        if not use_pseudoinverse:
            theta_updated,cost,t,delta,lamba = levenburgMarquardtStep(jacobian_matrix,jacobian_matrix_barrier,jacobian_matrix_full,r,I,lamba,delta,damping_multiplier,F,t,k,cost,alpha,datapoints,numberOfPoints)
        else:
            theta_updated,t,delta,cost = lineSearchStep(jacobian_matrix,jacobian_matrix_barrier,jacobian_matrix_full,r,I,lamba,delta,damping_multiplier,F,t,k,cost,alpha,datapoints,numberOfPoints,tolDelta,gamma)

        # Check if the latest update overshot the barrier term
        if (t[:,k+1]).dot((F).dot(t[:,k+1])) <=0:
            use_pseudoinverse = True
            lamba = 0
            t[:,k+1] = t[:,k]
            if k>0:
                t[:,k] = t[:,k-1]

        # Check for various stopping criteria to end the main loop
        elif min(np.linalg.norm(t[:,k+1]-t[:,k]),np.linalg.norm(t[:,k+1]+t[:,k])) < tolTheta and theta_updated:
            keep_going=False
        elif abs(cost[k] - cost[k+1]) < tolCost and theta_updated:
            keep_going=False
        elif np.linalg.norm(delta[:,k+1]) < tolDelta and theta_updated:
            keep_going=False
            
        k = k + 1
    #theta = t[:,k];
    theta = np.array([t[0,k],t[1,k],t[2,k],t[3,k],t[4,k],t[5,k]],dtype=np.float64)
    theta = theta/np.linalg.norm(theta)
    return theta

@jit(nopython=True)
def levenburgMarquardtStep(jacobian_matrix,jacobian_matrix_barrier,jacobian_matrix_full,r,I,lamba,delta,damping_multiplier,F,t,k,cost,alpha,datapoints,numberOfPoints):

    #jacobian_matrix = struct.jacobian_matrix;
    #jacobian_matrix_barrier = struct.jacobian_matrix_barrier;
    #r = struct.r;
    #I = struct.I;
    #lamba = struct.lamba;
    #delta = struct.delta[:,struct.k];
    #delta = struct.delta[struct.k];

    delta_current = delta[:,k]
    t_current = t[:,k]
    
    #damping_multiplier = struct.damping_multiplier;
    #F = struct.F;
    #t = struct.t[:,struct.k];
    #current_cost = struct.cost[struct.k];
    current_cost = cost[k]
    #alpha = struct.alpha;
    data_points = datapoints
    #data_points = datapoints;
    #numberOfPoints = struct.numberOfPoints;

    # compute two potential updates for theta based on different weightings of the identity matrix
    #########################

    jacob = (jacobian_matrix_full.transpose()).dot(r)

    tFt = t_current.dot(F.dot(t_current))

    # We solve for the new update direction in a numerically careful manner
    # If we didn't worry about numerical stability then we would compute 
    # the first new search direction like this:
    
    # update_a = - (H+lambda*I)\jacob;
    
    # But close to the barrier between ellipses and hyperbolas we may
    # experience numerical conditioning problems due to the nature of the
    # barrier term itself.  Hence we perform the calculation in a
    # numerically more stable way with

    Z_a = np.zeros((12,12),dtype=np.float64)
    Z_a[0:6,0:6]=(jacobian_matrix.transpose()).dot(jacobian_matrix)+lamba*I
    Z_a[0:6,6:12]=(tFt**4)*np.outer(jacobian_matrix_barrier,jacobian_matrix_barrier)
    Z_a[6:12,0:6]=I
    Z_a[6:12,6:12]=-(tFt**4)*I

    zz_a = np.zeros(12,dtype=np.float64)
    zz_a[0:6]= -jacob

    update_a = np.linalg.solve(Z_a, zz_a)
    # drop the nuisance parameter components
    update_a = update_a[0:6]

    # In a similar fashion, the second potential search direction could be
    # computed like this:
    
    # update_b = - (H+(lambda/v)*I)\jacob
    
    # but instead we computed it with

    Z_b = np.zeros((12,12),dtype=np.float64)
    Z_b[0:6,0:6]=(jacobian_matrix.transpose()).dot(jacobian_matrix)+(lamba/damping_multiplier)*I
    Z_b[0:6,6:12]=(tFt**4)*np.outer(jacobian_matrix_barrier,jacobian_matrix_barrier)
    Z_b[6:12,0:6]=I
    Z_b[6:12,6:12]=-(tFt**4)*I

    zz_b = np.zeros(12,dtype=np.float64)
    zz_b[0:6]= -jacob

    update_b = np.linalg.solve(Z_b, zz_b)
    # drop the nuisance parameter components
    update_b = update_b[0:6]

    # the potential new parameters are then 
    t_potential_a = t_current + update_a
    t_potential_b = t_current + update_b

    # compute new residuals and costs based on these updates
    ########################################################

    # residuals computed on data points
    cost_a = 0
    cost_b = 0
    for i in range(numberOfPoints):
        m = data_points[:,i]
        # transformed data point

        ux_j = np.array([m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1],dtype=np.float64)
        dux_j = np.array([[2*m[0],     m[1],     0,      1,    0,  0],[0,m[0],2*m[1],0,1,0]],dtype=np.float64)
        dux_j = dux_j.transpose()

        # outer product

        A = np.outer(ux_j,ux_j)

        B = dux_j.dot(dux_j.transpose())
        
        t_aBt_a = t_potential_a.dot(B.dot(t_potential_a))
        t_aAt_a = t_potential_a.dot(A.dot(t_potential_a))
        
        t_bBt_b = t_potential_b.dot(B.dot(t_potential_b))
        t_bAt_b = t_potential_b.dot(A.dot(t_potential_b))

        
        # AML cost for i'th data point
        cost_a = cost_a +  t_aAt_a/t_aBt_a 
        cost_b = cost_b +  t_bAt_b/t_bBt_b 
               
    # Barrier term
    t_aIt_a = t_potential_a.dot(I.dot(t_potential_a))
    t_aFt_a = t_potential_a.dot(F.dot(t_potential_a))
    
    t_bIt_b = t_potential_b.dot(I.dot(t_potential_b))
    t_bFt_b = t_potential_b.dot(F.dot(t_potential_b))

    # add the barrier term
    cost_a = cost_a + (alpha*(t_aIt_a/t_aFt_a))**2
    cost_b = cost_b + (alpha*(t_bIt_b/t_bFt_b))**2

    # determine appropriate damping and if possible select an update
    if cost_a >= current_cost and cost_b >= current_cost: 
        # neither update reduced the cost
        theta_updated = False
        # no change in the cost
        cost[k+1] = current_cost
        # no change in parameters
        t[:,k+1] = t_current
        # no changes in step direction

        #print("delta.shape",delta.shape)
        #print("struct.delta.shape",struct.delta.shape)
        
        delta[:,k+1] = delta_current
        # next iteration add more Identity matrix
        lamba = lamba * damping_multiplier
    elif cost_b < current_cost:
        # update 'b' reduced the cost function
        theta_updated = True
        # store the new cost
        cost[k+1] = cost_b
        # choose update 'b'
        t[:,k+1] = t_potential_b /np.linalg.norm(t_potential_b)
        # store the step direction
        delta[:,k+1] = update_b
        # next iteration add less Identity matrix
        lamba = lamba / damping_multiplier
    else:
        # update 'a' reduced the cost function
        theta_updated = True
        # store the new cost
        cost[k+1] = cost_a
        # choose update 'a'
        t[:,k+1] = t_potential_a / np.linalg.norm(t_potential_a)
        # store the step direction
        delta[:,k+1] = update_a
        # keep the same damping for the next iteration
        lamba = lamba

    # return a data structure containing all the updates
    return theta_updated,cost,t,delta,lamba

@jit(nopython=True)
def lineSearchStep(jacobian_matrix,jacobian_matrix_barrier,jacobian_matrix_full,r,I,lamba,delta,damping_multiplier,F,t,k,cost,alpha,datapoints,numberOfPoints,tolDelta,gamma):

    # extract variables from data structure
    ########################################
    t_current = t[:,k]
    #jacobian_matrix = struct.jacobian_matrix;
    #jacobian_matrix_barrier = struct.jacobian_matrix_barrier;
    #r = struct.r;
    #I = struct.I;
    #lamba = struct.lamba;
    #delta = struct.delta[struct.k];
    delta_current = delta[:,k]
    #tolDelta = struct.tolDelta;
    #damping_multiplier = struct.damping_multiplier;
    #F = struct.F;
    #I = struct.I;
    current_cost = cost[k]
    data_points = datapoints
    #alpha = struct.alpha;
    #gamma = struct.gamma;
    #numberOfPoints = struct.numberOfPoints;

    # jacobian (vector), r(t)'d/dtheta[r(t)]
    jacob = (jacobian_matrix_full.transpose()).dot(r)
    tFt = t_current.dot(F.dot(t_current))

    # We solve for the new update direction in a numerically careful manner
    # If we didn't worry about numerical stability then we would compute 
    # the first new search direction like this:

    # update = - pinv(H)*jacob;

    # But close to the barrier between ellipses and hyperbolas we may
    # experience numerical conditioning problems due to the nature of the
    # barrier term itself.  Hence we perform the calculation in a
    # numerically more stable way with 

    Z = np.zeros((12,12),dtype=np.float64)
    Z[0:6,0:6]=(jacobian_matrix.transpose()).dot(jacobian_matrix)+lamba*I
    Z[0:6,6:12]=(tFt**4)*np.outer(jacobian_matrix_barrier,jacobian_matrix_barrier)
    Z[6:12,0:6]=I
    Z[6:12,6:12]=-(tFt**4)*I

    zz = np.zeros((12),dtype=np.float64)
    zz[0:6]= -jacob

    update = (np.linalg.pinv(Z,1e-20)).dot(zz)
    # drop the nuisance parameter components
    update = update[0:6]

    # there is no repeat...until construct so we use a while-do
    frac = 0.5
    while True:
        # compute potential update    
        t_potential = t_current + frac*update
        delta_current = frac*update
        # halve the step-size
        frac = frac / 2 
        # compute new residuals on data points
        local_cost = 0
        for i in range(numberOfPoints):

            m = data_points[:,i]
            # transformed data point

            ux_j = np.array([m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1],dtype=np.float64)
            dux_j = np.array([[2*m[0],     m[1],     0,      1,    0,  0],[0,m[0],2*m[1],0,1,0]],dtype=np.float64)
            dux_j = dux_j.transpose()

            # outer product
            A = np.outer(ux_j,ux_j)

            B = dux_j.dot(dux_j.transpose())
            
            tBt = t_potential.dot(B.dot(t_potential))
            tAt = t_potential.dot(A.dot(t_potential))
                   
            # AML cost for i'th data point
            local_cost = local_cost +  tAt/tBt             
        

        # Barrier term
        tIt = t_potential.dot(I.dot(t_potential))
        tFt = t_potential.dot(F.dot(t_potential))

        # add the barrier term
        local_cost = local_cost + (alpha*(tIt/tFt))**2

        # check to see if cost function was sufficiently decreased, and whether
        # the estimate is still an ellipse. Additonally, if the step size
        # becomes too small we stop. 
        if t_potential.dot(F.dot(t_potential)) > 0 and (local_cost < (1-frac*gamma)*current_cost)  or np.linalg.norm(delta_current) < tolDelta:
            break
    


    theta_updated = True
    t[:,k+1] = t_potential / np.linalg.norm(t_potential)
    delta[:,k+1] = delta_current
    cost[k+1] = local_cost

    # return a data structure with all the updates
    return theta_updated,t,delta,cost

@jit(nopython=True)    
def orthogonalLineFit(points):
    # I'm assuming "points" is a N x 2 numpy array.
    (rows,cols)=points.shape
    if not cols ==2:
        raise ValueError('expected a N x 2 array')
        #logging.error('expected a N x 2 array')

    num_points = rows

    #c_x = sum(points[:,0])/num_points;
    #c_y = sum(points[:,1])/num_points;

    # using a for loop instead of "sum" makes @jit happy.
    c_x=0
    c_y=0
    for j in range(num_points):
        c_x=c_x+points[j,0]
        c_y=c_y+points[j,1]
    c_x = c_x/num_points
    c_y = c_y/num_points
    
    #center = c_x+1j*c_y;

    #points_complex = np.zeros(rows);
    #points_complex = points[:,0]+(1j)*points[:,1];

    #center2 = sum(points_complex)/num_points;
    #Z = sum((points_complex-center)**2);

    Z_r = 0
    Z_i = 0
    for j in range(num_points):
        #Z = Z + (points_complex[j]-center)**2;
        Z_r = Z_r+(points[j,0]-c_x)**2-(points[j,1]-c_y)**2
        Z_i = Z_i+2*(points[j,1]-c_y)*(points[j,0]-c_x)
    
    #if Z==0:
    if Z_r == 0 and Z_i == 0:
        raise ValueError('the line is a point')
        #logging.error('the line is a point')
    else:
        #Z = Z**0.5;
        Z_sqrt_r = math.sqrt((Z_r+math.sqrt(Z_r**2+Z_i**2))/2)
        Z_sqrt_i = math.sqrt((-Z_r+math.sqrt(Z_r**2+Z_i**2))/2)
        if Z_i<0:
            Z_sqrt_i = -Z_sqrt_i
        # p and v are the vector form of the line l(t) = p+t*v
        #p = np.array([center.real,center.imag]);
        #v = np.array([Z.real,Z.imag])/abs(Z); # make it a unit vector
        p = np.array([c_x,c_y])
        v = np.array([Z_sqrt_r,Z_sqrt_i])
        v = v/np.linalg.norm(v)
        # a,b,c are the implicit form a*x+b*y+c=0
        a = -v[1]
        b = v[0]
        c = -(-v[1]*p[0]+v[0]*p[1])

        equation = np.array([0,0,0,a,b,c])

        error = np.zeros(rows)
        for k in range(rows):
            error[k]=np.linalg.norm((points[k,:]-p)-((points[k,:]-p).dot(v))*v)
        
    return equation,error
    
