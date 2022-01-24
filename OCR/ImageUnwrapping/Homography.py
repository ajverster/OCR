import numpy as np;
import math;
from math import *
import numba
from numba import jit

# routine for computing the unique homography mapping the four input points p_i
# to the four output points P_i.  This routine will break down if any three of either p_i or P_i are colinear.
@jit(nopython=True)
def ComputeHomographyFromPoints(p1,p2,p3,p4,P1,P2,P3,P4):
    # we want p_i in the old image to map to P_i in the new image.

    # the formula for this comes straight out of "Multiple View Geometry,
    # 2nd Edition", page 34, example 2.12.

    # each point mapping (x,y) -> (x',y') gives two rows of the form
    # [ -x -y -1 0 0 0 x'x x'y x';
    # [ 0 0 0 -x -y -1 y'x y'y y']

    M = np.zeros((9,9)) # 9x9 linear system to solve for the 3x3 matrix H 
                         # which is represented as a colum vector


    # first 8 rows come from point constraints.

    for k in range(4):
        if k==0:
            
            x = p1[0]
            y = p1[1]
            xp = P1[0]
            yp = P1[1]


        elif k==1:
            
            x = p2[0]
            y = p2[1]
            xp = P2[0]
            yp = P2[1]


        elif k==2:
            
            x = p3[0]
            y = p3[1]
            xp = P3[0]
            yp = P3[1]


        else:
            
            x = p4[0]
            y = p4[1]
            xp = P4[0]
            yp = P4[1]

        M[2*k,:] =np.array([ -x, -y, -1,  0,  0,  0, xp*x, xp*y, xp])
        M[2*k+1,:]=np.array([ 0,  0,  0, -x, -y, -1, yp*x, yp*y, yp])

    # actually, the final matrix H is unique up to a scaling.  The final
    # row fixes the scaling.
    # since H is invertible, it must have at least one non-zero entry.  Our
    # constraint will be setting some specific entree equal to 1.  However,
    # it may be that certain entrees have to be 0, so we don't know in
    # advance which entree we can use.  We'll cycle through them and stop
    # when we get an invertible matrix.
    k = 0
    while np.linalg.matrix_rank(M)<9 and k<9:
        M[8,:]=np.zeros(9)
        M[8,k]=1
        k = k+1

    # rhs is 0 except for the final entree
    b = np.zeros(9)
    b[8]=1

    if np.linalg.matrix_rank(M)<9:
        H = None
        success = False
    else:
        h = np.linalg.solve(M,b);
        success = True
        for k in range(9):
            if not isfinite(h[k]):
                success=False


        H = np.array([[h[0], h[1], h[2]],[h[3], h[4], h[5]],[h[6], h[7], h[8]]])
    return H,success

# We want to apply the homography H above to our image I.  To do this, we need a bounding box H(I) - otherwise we
# don't know how many rows and columns to give our new image.  To find this, we take advantage of the fact that homographies
# take lines to lines.  since I is a rectangle with corners p1,p2,p3,p4, H(I) will be a diamond with corners H(p1), H(p2), H(p3), H(p4).
# These corners give us our bounding box.
def getBoundingBoxFromHomography(oldImageWidth,oldImageHeight,H):

    # these four points bound our original image, in homogenous
    # coordinates.

    p1 = np.array([0,0,1])
    p2 = np.array([oldImageWidth-1,0,1])
    p3 = np.array([0,oldImageHeight-1,1])
    p4 = np.array([oldImageWidth-1,oldImageHeight-1,1])

    P1 = H.dot(p1)
    P2 = H.dot(p2)
    P3 = H.dot(p3)
    P4 = H.dot(p4)

    # make sure none of the points are mapped to infinity
    if P1[2]!=0 and P2[2]!=0 and P3[2]!=0 and P4[2]!=0:
        x_min = min([P1[0]/P1[2], P2[0]/P2[2], P3[0]/P3[2], P4[0]/P4[2]])
        x_max = max([P1[0]/P1[2], P2[0]/P2[2], P3[0]/P3[2], P4[0]/P4[2]])
        y_min = min([P1[1]/P1[2], P2[1]/P2[2], P3[1]/P3[2], P4[1]/P4[2]])
        y_max = max([P1[1]/P1[2], P2[1]/P2[2], P3[1]/P3[2], P4[1]/P4[2]])
        success = True
    else:
        x_min=None
        x_max=None
        y_min=None
        y_max=None
        success=False
    return x_min,x_max,y_min,y_max,success

# here we actually compute H(I), using the bounding box above.  This routine is expensive, so it is
# accelerated using numba.
@jit(nopython=True)
def applyHomographyToImageUsingBoundingBox(H,I,x_min,x_max,y_min,y_max,inpaintFlag):

    # we want to compute the image I2 formed by applying H to I.
    # to do this, for each pixel in I2, we using inv(H) to find out where
    # it came from in I.  We will land between pixel centers, and hence
    # must apply bilinear interpolation.

    # some of the pixels visible after applying the homography will not have
    # been invisible in the original image.  We color them black.

    H_back = np.linalg.inv(H); # this is ok because the matrix is only 3x3

    n,m,num_channels=I.shape

    NEW_WIDTH =ceil(x_max-x_min)
    NEW_HEIGHT = ceil(y_max-y_min)

    mask = np.zeros((NEW_HEIGHT,NEW_WIDTH),np.uint8);

    if not isfinite(NEW_WIDTH) or not isfinite(NEW_HEIGHT):
        success = False
        return I,success,mask
    else:

        I2 = np.zeros((NEW_HEIGHT,NEW_WIDTH,num_channels),np.uint8)

        for j in range(NEW_HEIGHT):
            for k in range(NEW_WIDTH):

                # we need to translate our coordinate system so that pixel coordinates (0,0) get mapped to
                # physical coordinates (x_min,y_min).  Remember also that pixel coordinates (j,k) go column-row
                # which is the opposite of cartesian coordinates (x,y).
                
                x = x_min+k
                y = y_min+j
                
                xp = (H_back[0,0]*x+H_back[0,1]*y+H_back[0,2])/(H_back[2,0]*x+H_back[2,1]*y+H_back[2,2])
                 
                yp = (H_back[1,0]*x+H_back[1,1]*y+H_back[1,2])/(H_back[2,0]*x+H_back[2,1]*y+H_back[2,2])

                # again, the same annoying swap between (j,k) image coordinates
                # and x,y image coordinates.
                 
                jp = yp
                kp = xp
                 
                
                # now we do bilinear interpolation.
                K = floor(kp)
                J = floor(jp)

                s = kp-K;
                t = jp-J;

                
                # if inpaintFlag == 0, then no inpainting is done and no special boundary conditions are applied.
                
                if inpaintFlag == 1: # periodic boundary conditions
                    K = K%(m-1)
                    J = J%(n-1)
                elif inpaintFlag == 2: # clamping
                    if K<0:
                        K = 0
                        s = 0
                    if K>m-2:
                        K = m-2
                        s = 1
                    if J<0:
                        J = 0
                        t = 0
                    if J> n-2:
                        J = n-2
                        t = 1
                # if inpaintFlag == 3 or 4, we're going to inpaint.  We'll build a mask for that purpose shortly.

                # make sure we have landed somewhere sensible.

                if 0<=K and 0<=J and K<m-1 and J<n-1:
                    # go channel by channel;
                    for l in range(num_channels):
                        I2[j,k,l]=I[J,K,l]*(1-s)*(1-t)+I[J+1,K+1,l]*s*t+I[J,K+1,l]*s*(1-t)+I[J+1,K,l]*(1-s)*t
                else:
                    if inpaintFlag == 3 or inpaintFlag == 4:  # we only have to construct the mask if we are doing inpainting
                        mask[j,k]=1
        success = True

        return I2,success,mask
