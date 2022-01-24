import numpy as np
import OCR.ImageUnwrapping.Ellipse as Ellipse
import math
from math import *
import logging

# The curves relevant to us are lines and ellipses.  I could have had a superclass "Curve",
# and then made "Ellipse" and "Line" inherit from it.  However, I decided to just have one
# class, with a flag that tells you if you are dealing with a line or an Ellipse.

# One reason for doing this is that it happens to be particularly easy to describe lines and ellipses
# within a unified framework.  For one thing, they are both conics, that is curves of the form

# Ax^2+Bxy+Cy^2+Dx+Ey+F = 0    (1)

# For lines, we have A = B = C = 0.  For ellipses, we have B^2-4AC<0 (and hence A,C>0).
# The above coefficients are stored in a numpy array that I have called "equation", that is
# self.equation = np.array([A,B,C,D,E,F]);

# Another reason for doing this in a single class is that a line can be thought of as a special case
# of an ellipse - where the minor and major axises are 0 and infinity respectively.

# So, while there are some if statements of the form

# if self.isEllipse:
#   <do something>
# else:
#   <do something else>

# much of the code is reausable, and in fact could be applied not only to lines and ellipses
#but to ANY conic - what matters is that we are a curve of the form (1).

# To be a little bit more precise - the curves we are dealing with will never be full ellipses,
# rather they will be partial elliptic arcs.  Because of this, it makes sense to talk about the
# "concavity" of the ellipse, a concept that would otherwise be meaningless.  The concavity of
# Our elliptic arcs can take on the values "up", "down", "leftwards", or "rightwards" and is
# critical for making inferences about the nature of the image we are working with
# (lines have concavity "none").

# Also because we are dealing with elliptic arcs rather than full ellipses, it makes sense to talk
# about an "extreme point" of the arc - this is the point where the tangent line is either horizontal
# or vertical.  In our situation, only one extreme point should ever be present.

# the extreme point is in turn critical because we use it for quantifying the curvature of our ellipse,
# which is defined to be the second derivative evaluated at the extreme point.  

class Curve(object):
    def __init__(self,pixels,equation,error_vector,isEllipse,im_width,im_height):
        # pixels should be N x 2, not 2 x N.
        self.pixels = pixels
        self.equation = equation
        self.isEllipse = isEllipse
        self.im_width = im_width
        self.im_height = im_height
        self.error_vector = error_vector
        concavity,x_extremal = self.getConcavity()
        self.concavity = concavity
        self.x_extremal = x_extremal
        self.confidence = self.getConfidence()
        self.minor_axis_length,self.major_axis_length =self.get_axis_length()
        horiz_comp,vert_comp=self.analyze_orientation()
        self.horizontally_complete = horiz_comp
        self.vertically_complete = vert_comp
        # if the curve is line, it's useful to know it's intersection with the edge of the image.
        # if it's an ellipse, this is set to None.
        self.endPoints = self.getEndPoints()
        self.curvature = self.getCurvature()

    def getCurvature(self):
        # if a curve is a line, it obviously has zero curvature.
        # if it is an ellipse, there is no single curvature - it will
        # vary from point to point.  However, I am *defining* the curvature
        # ellipse to be its curvature at a convenient reference point -
        # namely the extremal point of the ellipse
        # (that is, the point where the tangent line is horizontal or vertical).
        # Here, the curvature is the same as the second derivative of the parametrization
        # of the ellipse as
        # y = f(x) (horizontally complete case)
        # or x=f(y) (vertically complete case).
        # If the ellipse is neither vertically complete nor horizontally complete
        # we don't have the above parametrization - in this case 
        # I set the curvature equal to None.
        # This is fine, because all such curves will be discarded and never make it
        # into the image skeleton.
        if not self.isEllipse:
            return 0
        else:
            A=self.equation[0]
            B=self.equation[1]
            C=self.equation[2]
            D=self.equation[3]
            E=self.equation[4]
            F=self.equation[5]
            if self.horizontally_complete:
                return -2*A/(B*self.x_extremal[0]+2*C*self.x_extremal[1]+E)
            elif self.vertically_complete:
                # x <-> y
                # A <-> C
                # D <-> E
                # B,F fixed.
                return -2*C/(B*self.x_extremal[1]+2*A*self.x_extremal[0]+D)
            return None

    def printInfo(self):
        
        logging.debug("equation = ",self.equation);
        logging.debug("isEllipse =",self.isEllipse);
        logging.debug("concavity =",self.concavity);
        logging.debug("confidence =",self.confidence);
        logging.debug("horizontally, vertically complete =",self.horizontally_complete,self.vertically_complete );
        logging.debug("minor and major axis length =",self.minor_axis_length,self.major_axis_length);
        logging.debug("x_extremal = ",self.x_extremal);
        logging.debug("curvature = ",self.curvature);

        return
        
    def getConfidence(self):

        # basically, we are assigning confidence +1 to pixels within distance delta
        # of the curve, and confidence -1 to pixels distance greater than delta
        # and the overall confidence will be the sum over pixelwise confidences.
        # however, instead of discrete +/-1, we smoothingly interpolate using
        # bellcurve.
        
        delta = 3
        
        C=np.zeros(len(self.error_vector))
        for j in range(len(self.error_vector)):
            C[j]=-1+2*exp(-(self.error_vector[j]**2)*log(2)/(delta^2))
        return np.sum(C)
    
    def getConcavity(self):
       
            if self.isEllipse:
                
                
                p_top,p_bottom,p_left,p_right=Ellipse.getEllipseExtremePoints(self.equation)
                
                # is p_top or p_bottom closest to the pixels making up the
                # ellipse?
                d_top = min((self.pixels[:,0]-p_top[0])**2+(self.pixels[:,1]-p_top[1])**2)
                d_bottom = min((self.pixels[:,0]-p_bottom[0])**2+(self.pixels[:,1]-p_bottom[1])**2)
                d_left = min((self.pixels[:,0]-p_left[0])**2+(self.pixels[:,1]-p_left[1])**2)
                d_right = min((self.pixels[:,0]-p_right[0])**2+(self.pixels[:,1]-p_right[1])**2)
                d_star = min([d_top,d_bottom,d_left,d_right])
                if d_top == d_star:
                    concavity = 'down'
                    x_extremal = p_top
                elif d_bottom == d_star:
                    concavity = 'up'
                    x_extremal = p_bottom
                elif d_left == d_star:
                    concavity = 'rightwards'
                    x_extremal = p_left
                else:
                    concavity = 'leftwards'
                    x_extremal = p_right
                
            else:
                # lines aren't convex or concave
                concavity = 'none'
                x_mid = floor(self.im_width/2)
                y_mid = floor(self.im_height/2)
                X = self.equation
                a =X[3]; b=X[4]; c=X[5]; # ax+by+c = 0
                v = np.array([-b,a]) # vector parallel to the line.
                v = v/np.linalg.norm(v)
                phase = atan2(v[1],v[0])
                # Are we in a cone making an angle of +/- pi/4 with the x-axis?  If so, we're "mostly horizontal".
                # In this case, we define x_extremal as the point on the line with x-coordinate equal to one half of
                # the image width.  Otherwise, we define it as the point on the line with y-coordinate equal to one half of
                # the image height
                if (-pi/4<= phase and phase <=pi/4) or phase > 3*pi/4 or phase < -3*pi/4:
                    y = -(c+a*x_mid)/b
                    x_extremal = np.array([x_mid,y])
                else:
                    x = -(c+b*y_mid)/a
                    x_extremal = np.array([x,y_mid])
            return concavity,x_extremal

    def get_axis_length(self):
        if not self.isEllipse:
           b = 0
           a = inf
        else:
           a,b,x0,y0,phi=Ellipse.convertEllipseToGeometricForm(self.equation)
        return b,a


    def getEndPoints(self):
        # if our curve is a line, it is useful to know its endpoints in order to draw it.
        # if it is an ellipse, even though it is still meaningful to talk about endpoints,
        # it isn't useful, so we return None
        if self.isEllipse:
            return None;
        else:
            X = self.equation
            a=X[3];b=X[4];c=X[5] # ax+by+c=0 form of the line

            left_int = np.cross(np.array([a,b,c]),np.array([1,0,0]))
            right_int = np.cross(np.array([a,b,c]),np.array([1,0,-(self.im_width-1)]))
            bottom_int = np.cross(np.array([a,b,c]),np.array([0,1,0]))
            top_int = np.cross(np.array([a,b,c]),np.array([0,1,-(self.im_height-1)]))

            endPoints = []

            if left_int[2]!=0 and right_int[2]!=0: # this means we are NOT parallel to x = constant lines
               y_left = left_int[1]/left_int[2]
               y_right = right_int[1]/right_int[2]
               if 0<=y_left and y_left <= self.im_height-1:
                    endPoints.append(np.array([0,y_left]))
               if 0<= y_right and y_right <= self.im_height-1:
                   endPoints.append(np.array([self.im_width-1,y_right]))


            if bottom_int[2]!=0 and top_int[2]!=0: # this means we are NOT parallel to y = constant lines
               x_bottom = bottom_int[0]/bottom_int[2]
               x_top = top_int[0]/top_int[2]
               if 0<=x_bottom and x_bottom <= self.im_width-1:
                   endPoints.append(np.array([x_bottom,0]))
               if 0<=x_top and x_top <= self.im_width-1:
                   endPoints.append(np.array([x_top,self.im_height-1]))
            return endPoints;


    def analyze_orientation(self):
        if not self.isEllipse:
           
           # lines are horizontally complete if they are not of the form x=constant.
           # lines are vertically complete if they are not of the form y = constant.
           # a line can be both vertically and horizontally complete (in fact, most of the time it will be).
           
           # From "Multiple View Geometry in Computer Vision, Second
           # Edition", page 27, we can represent the intersection of
           # two lines of the form l1 = a1x+b1y+c1 ~ [a1,b1,c1] 
           # and l2 = [a2,b2,c2] as the cross product l1 x l2.
           
           # after computing the cross product (x,y,z) = l1 x l2, you
           # divide x and y by z to produce the inhomogeneous
           # representation of the point.  However, if z=0, this means
           # the intersection is at infinity, which means the lines are
           # parallel.
           
           X = self.equation
           a=X[3];b=X[4];c=X[5] # ax+by+c=0 form of the line
           
           left_int = np.cross(np.array([a,b,c]),np.array([1,0,0]))
           right_int = np.cross(np.array([a,b,c]),np.array([1,0,-(self.im_width-1)]))
           bottom_int = np.cross(np.array([a,b,c]),np.array([0,1,0]))
           top_int = np.cross(np.array([a,b,c]),np.array([0,1,-(self.im_height-1)]))
           
           if left_int[2]!=0 and right_int[2]!=0: # this means we are NOT parallel to x = constant lines

               y_left = left_int[1]/left_int[2]
               y_right = right_int[1]/right_int[2]
               if 0<=y_left and 0<= y_right and y_left <= self.im_height-1 and y_left <= self.im_height-1:
                   horiz_comp=True
               else:
                   horiz_comp=False
           else:
               horiz_comp=False
           
           if bottom_int[2]!=0 and top_int[2]!=0: # this means we are NOT parallel to y = constant lines

               x_bottom = bottom_int[0]/bottom_int[2]
               x_top = top_int[0]/top_int[2]
               if 0<=x_bottom and 0<= x_top and x_bottom <= self.im_width-1 and x_top <= self.im_width-1:
                   vert_comp=True
               else:
                   vert_comp=False
           else:
               vert_comp=False
           

           
           
        else:
            
           p_top,p_bottom,p_left,p_right=Ellipse.getEllipseExtremePoints(self.equation)
           
           if p_left[0]<0 and p_right[0]>self.im_width-1:
               horiz_comp = True
           else:
               horiz_comp = False
           
           if p_bottom[1]<0 and p_top[1]>self.im_height-1:
               vert_comp = True
           else:
               vert_comp = False
        return horiz_comp,vert_comp

    def isRedundant(self,otherCurve):

        if self.horizontally_complete and otherCurve.horizontally_complete:
            # in this case we can represent each curve as a function y = f(x).
            # compute min_x |f1(x)-f2(x)|.  If this min is less than a tolerance,
            # we declare the curves to be redundant.

            minVerticalDistance = inf

            tol = 15 # distance in pixels
        
            for x in range(self.im_width):
                
                y1 = self.get_y(x)
                y2 = otherCurve.get_y(x)
                if not y1==None and not y2==None:
                    if abs(y1-y2)<minVerticalDistance:
                        minVerticalDistance = abs(y1-y2)
            if minVerticalDistance<tol:
                return True
            else:
                return False
        if self.vertically_complete and otherCurve.vertically_complete:
            # in this case we can represent each curve as a function x = f(y).
            # compute min_y |f1(y)-f2(y)|.  If this min is less than a tolerance,
            # we declare the curves to be redundant.

            minHorizontalDistance = inf

            tol = 15
        
            for y in range(self.im_height):
                
                x1 = self.get_x(y)
                x2 = otherCurve.get_x(y)
                if not x1==None and not x2==None:
                    if abs(x1-x2)<minHorizontalDistance:
                        minHorizontalDistance = abs(x1-x2)
            if minHorizontalDistance<tol:
                return True
            else:
                return False

        # if we have gotten here, then they aren't redundant
        return False

    def get_y(self,x):

        A=self.equation[0]
        B=self.equation[1]
        C=self.equation[2]
        D=self.equation[3]
        E=self.equation[4]
        F=self.equation[5]
       
        if self.isEllipse:
            
            a = C
            b = (E+B*x)
            c = (A*x**2+D*x+F)
            
            desc = b**2-4*a*c
            if desc < 0:
                logging.debug('discrimant negative! desc = ',desc)
                y = None
            else:
                y1 = (-b+sqrt(b**2-4*a*c))/(2*a);
                y2 = (-b-sqrt(b**2-4*a*c))/(2*a);
                if self.concavity=='down':
                    y = max(y1,y2)
                elif self.concavity=='up':
                    y = min(y1,y2)
                else:
                    y = None
        else:
            if E!=0:
                y = -(D*x+F)/E
            else:
                y = None
        return y

    def get_x(self,y):

        A=self.equation[0]
        B=self.equation[1]
        C=self.equation[2]
        D=self.equation[3]
        E=self.equation[4]
        F=self.equation[5]
       
        if self.isEllipse:

        # x <-> y
        # A <-> C
        # D <-> E
        # B,F fixed.
            
            a = C
            b = (D+B*y)
            c = (C*y**2+E*y+F)
            
            desc = b**2-4*a*c
            if desc < 0:
                logging.debug('discrimant negative! desc = ',desc)
                x = None
            else:
                x1 = (-b+sqrt(b**2-4*a*c))/(2*a)
                x2 = (-b-sqrt(b**2-4*a*c))/(2*a)
                if self.concavity=='leftwards':
                    x = max(x1,x2)
                elif self.concavity=='rightwards':
                    x = min(x1,x2)
                else:
                    x = None
        else:
            if D!=0:
                x = -(E*y+F)/D
            else:
                x = None
        return x

    def flip(self):
        X = self.equation;A=X[0];B=X[1];C=X[2];D=X[3];E=X[4];F=X[5]
        Anew = C
        Bnew = B
        Cnew = A
        Dnew = E
        Enew = D
        Fnew = F
        newEquation = np.array([Anew,Bnew,Cnew,Dnew,Enew,Fnew])
        Pixels = self.pixels
        newPixels=np.zeros_like(Pixels)
        newPixels[:,0]=Pixels[:,1]
        newPixels[:,1]=Pixels[:,0]
        new_image_width = self.im_height
        new_image_height = self.im_width

        flippedCurve = Curve(newPixels,newEquation,self.error_vector,self.isEllipse,new_image_width,new_image_height)
        return flippedCurve

    def isUseful(self):
        if self.isEllipse:
            if self.horizontally_complete and self.confidence > 0.5*self.im_width:                
                return True
            else:
                return False
                
        else:
            # most lines will be both vertically and horizontally complete.
            # We want the line to be "close-ish" to horizontal.  We'll say anything
            # with a slope of at most 45 degrees qualifies.
            
            X = self.equation;A=X[0];B=X[1];C=X[2];D=X[3];E=X[4];F=X[5]
            v = np.array([-E,D])/sqrt(E**2+D**2) # this is the tangent vector to the line.
            theta = abs(asin(np.linalg.norm(np.cross(np.array([v[0],v[1],0]),np.array([1,0,0])))))
            #if theta < pi/4 and self.confidence > 0.5*self.im_width:
            if theta < pi and self.confidence > 0.5*self.im_width:
                return True
            else:
                return False

    def getIntersectionWithLine(self,line):

        if not self.isEllipse:

            # compute line-line intersection.
            X=self.equation; a=X[3];b=X[4];c=X[5]
            X_line = line.equation; a_l = X_line[3];b_l = X_line[4];c_l = X_line[5]

            # I'm using the line-line intersection formula given in Result 2.2 on page 27
            # Of "Multiple View Geometry in Computer Vision, Second Edition"
            
            p = np.cross(np.array([a,b,c,]),np.array([a_l,b_l,c_l]))

            if p[2]==0: # this means the lines are parallel
                p_int = None
                int_found = False
            else:
                p_int = np.array([p[0]/p[2],p[1]/p[2]])
                int_found = True
        else:
            # convert the line to the form l(t) = p+t*v
            X_line = line.equation; a_l = X_line[3];b_l = X_line[4];c_l = X_line[5]
            v = np.array([-b_l,a_l])/sqrt(b_l**2+a_l**2)
            # p will either be x or y intercept.
            if abs(a_l)>abs(b_l):
                p = np.array([-c_l/a_l,0])
            else:
                p = np.array([0,-c_l/b_l])

            p_int,int_found = Ellipse.get_intersection_with_line(self.equation,p,v,self.concavity);
        return p_int,int_found

    # returns True if and only if the curve is a line making an angle of at most
    # 1/100 of a degree with the y-axis.
    def isVerticalLine(self):
        if self.isEllipse:
            return False
        else:
            X=self.equation; a=X[3];b=X[4];c=X[5]
            v = np.array([-b,a,0])/sqrt(a**2+b**2) # tangent to line.
            angle_with_y_axis = abs(asin(np.linalg.norm(np.cross(np.array([0,1,0]),v))))
            angle_with_y_axis=angle_with_y_axis*180/pi # convert to degrees.
            if angle_with_y_axis < 0.01: # we allow a 0.01 degree error.
                return True;
            else:
                return False;

    def applyHomography_and_translation(self,H,x0,y0,newWidth,newHeight):
        X = self.equation/np.linalg.norm(self.equation)

        if self.isEllipse:
        
            A=X[0];B=X[1];C=X[2];D=X[3];E=X[4];F=X[5]

            # first we apply the homography

            # I'm using the transformation equation (result 2.13)
            # on page 37 of "Multiple View Geometry Second Edition"
            # for the transformation of conics under a homography.

            M = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]],dtype=np.float64)

            H_inv = np.linalg.inv(H)
             
            M = (H_inv.transpose()).dot(M).dot(H_inv)

            A =  M[0,0]; B = 2*M[1,0]; C = M[1,1]
            D =2*M[0,2]; E = 2*M[1,2]; F = M[2,2]

            # next we apply the translation x <- x+x0, y <-y+y0

            A = A; B = B; C = C
            F = F+A*x0**2+B*x0*y0+C*y0**2-x0*D-y0*E
            D = D-2*x0*A-y0*B
            E = E-2*y0*C-x0*B


            newEquation = np.array([A,B,C,D,E,F])

        else:

            # in principle, since a line is a special type of conic, we should
            # be able to apply exactly the same formula as above.  Indeed, in the
            # Matlab prototype, this was fine.  However, there is a more specific
            # equation for transformation of lines "Multiple view Geometry Second Edition"
            # page 36 equation (2.6).  For whatever reason, this gives accurate results
            # whereas the general formula yields slight errors.
            
            a=X[3];b=X[4];c=X[5]
            H_inv = np.linalg.inv(H)
            l = np.array([a,b,c],dtype=np.float64)
            l_new = (H_inv.transpose()).dot(l)
            a_new = l_new[0];b_new = l_new[1]; c_new = l_new[2]

            c_new = c_new-x0*a_new-y0*b_new

            newEquation = np.array([0,0,0,a_new,b_new,c_new])
            

        # now, to transform the points.  

        Pixels = self.pixels
        num_pixels,_=Pixels.shape


        #First, to do the homography, we first add a column of ones.

        PixelsHomogenous = np.zeros((num_pixels,3))
        PixelsHomogenous[:,0]=Pixels[:,0]
        PixelsHomogenous[:,1]=Pixels[:,1]
        PixelsHomogenous[:,2]=np.ones(num_pixels)

        newPixelsHomogenous = (H.dot(PixelsHomogenous.transpose())).transpose()
        newPixels = np.zeros((num_pixels,2))
        newPixels[:,0] = newPixelsHomogenous[:,0]/newPixelsHomogenous[:,2] #elementwise division
        newPixels[:,1] = newPixelsHomogenous[:,1]/newPixelsHomogenous[:,2] #elementwise division
        #Next, let's translate

        newPixels[:,0]=newPixels[:,0]+x0
        newPixels[:,1]=newPixels[:,1]+y0

        newCurve = Curve(newPixels,newEquation,self.error_vector,self.isEllipse,newWidth,newHeight)
        return newCurve
            

            
            

    



