# import external libraries
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging
import copy

#import my own libraries
import OCR.ImageUnwrapping.curve as curve
import OCR.ImageUnwrapping.curveFitOptimized as curveFitOptimized
import OCR.ImageUnwrapping.Ellipse as Ellipse
import OCR.ImageUnwrapping.skeletonHelperFunctions as skeletonHelperFunctions
import OCR.ImageUnwrapping.Homography as Homography


class Skeleton(object):
    def __init__(self,image,debugFolderName =".",debug=False,inpaintFlag=False):

        # the skeleton will contain a list of curves, which is initialized to empty.
        # It also carries a copy of the image to be modified, which gets modified in steps, along
        # with the skeleton itself.  So for example, the function "perspectiveCorrection" consists of
        # two sub-steps - one which modifies skeleton.image and the other which modifies skeleton.curves

        # In order for the Skeleton to know whether or not it should be saving the results of certain sub-calculations,
        # It also is fed a copy of "debug" as well as the name of the folder in which intermediate images are saved.

        self.curves = []
        self.image = image
        self.debug = debug
        self.debugFolderName = debugFolderName
        self.inpaintFlag = inpaintFlag

        # later on we will have a field called "self.sortedUsefulCurves" and "self.y_vec" - but these don't exist yet.
        # They will be created when the function "getSortedUsefulCurves" is called.

        # In order to divide our image into meaningful curves, we will need a copy of the image
        # gradient.  This routine also computes statistics on the gradient  - the mean and variance
        # of the gradient norm - which are not currently used.  However, they could later possibly be useful
        # in determining an adaptive threshold for Canny-Edge detection, and the loss in performance is
        # negligible.  So, I haven't disabled their calculation.
        
        image_grad,meanNorm,varNorm = skeletonHelperFunctions.getImageGradient(image)

        # I'm currently logging this information.
        logging.debug("meanNorm, varNorm**0.5 = ",meanNorm,varNorm**0.5)

        # Next, we compute an edgeMap of the image using Canny edge detection, using constant upper and
        # lower thresholds t1 = 2 and t2 = 200.  Using constant thresholds is less than ideal, but seems
        # adaquete for our purposes.  Latter, these numbers could be calculated in terms of the gradient statistics
        # above which are currently ignored.

        edgeMap = cv2.Canny(image,2,200,L2gradient=True)

        if self.debug > 0:
            debug_path = self.debugFolderName+"/edgeMap.png";
            cv2.imwrite(debug_path,edgeMap)

        rows,cols,channels = image.shape

        # I'm now splitting the edge map into connected components (called "features" in the accompanying PDF).
        # The "features" themselves will later be subdivided into what should be basic curves, which will then
        # be fed into a curve-fit routine.  I'm going to use the words "feature", "component" and "fragment"
        # interchangeably.

        num_labels, labels, stats, centriods = cv2.connectedComponentsWithStats(edgeMap)

        # I'm storing the pixel coordinates of these connected components
        # (features) in a list of lists.  I'm using this data structure instead of
        # a numpy array because the number of components in each feature is different.

        # Initialize a list of num_labels many empty lists

        coordinates = [];
        
        for j in range(num_labels):
            coordinates.append([])

        # at the same time we initialize a numpy array to store the number of pixels
        # in each feature.
        fragmentSizes = np.zeros(num_labels)

        rows,cols = edgeMap.shape

        # this is a loop that puts the points making up each connected
        # component of the edgemap into a list of lists, while at the same
        # time computing the number of pixels in each component and storing it
        # in the numpy array "fragmentSizes"

        for i in range(rows):
            for j in range(cols):
                if labels[i][j]>0:
                    coordinates[labels[i][j]].append([j,i])
                    fragmentSizes[labels[i][j]]=fragmentSizes[labels[i][j]]+1
        
        fragmentMap = np.zeros_like(edgeMap)

        # Shortly, we will begin looping over the fragments, throwing away the ones
        # that are too small (in terms of either their width or height, or in terms
        # of the number of pixels they contain).  The fragments that don't get thrown
        # away will be subdivided into sub-fragments, by applying k-means clustering
        # their image gradients.


        # minimum fraction or the width or height of the image for a fragment to not be
        # thrown away.
        minFrac = 0.25 

        # create a progress bar as skeleton construction is the most time-consuming
        # part of the algorithm
        
        with tqdm.tqdm(total=num_labels, disable=True) as pbar:
            pbar.set_description("building skeleton...")
            # start from k=1 because k=0 is the label for the background.
            for k in range(1,num_labels):
                pbar.update(1)

                box_width = stats[k,cv2.CC_STAT_WIDTH]
                box_height = stats[k,cv2.CC_STAT_HEIGHT]
                num_pixels = fragmentSizes[k]

                # as I said, we throw away components that are too small or don't contain enough pixels.
                if ( box_width>cols*minFrac or box_height>rows*minFrac ) and num_pixels>minFrac*max(rows,cols) :

                    # Each feature/component/fragment that passes now gets split into sub-features, using clustering.
                    # The hope is that each sub-feature will be a simple curve.  We're going to loop through the sub-features,
                    # throwing away ones that are too small or don't contain enough pixels, and trying to fit curves to the ones
                    # that pass.
                    clusters=skeletonHelperFunctions.splitFragmentIntoClusters(coordinates[k],image_grad,self.debug)
                    for l in range(len(clusters)):
                        cluster = clusters[l]
                        cluster_box,cluster_size = skeletonHelperFunctions.getBoundingBoxAndSize(cluster)
                        box_width = cluster_box[2]
                        box_height = cluster_box[3]
                        
                        if ( box_width>cols*minFrac or box_height>rows*minFrac ) and cluster_size>minFrac*max(rows,cols):

                            # First try fitting a line.
                            lineEquation,lineError = curveFitOptimized.orthogonalLineFit(cluster)

                            # The idea is to now do an ellipse fit and see which fit is better.  However,
                            # if the line fit is very good - if the maximum error is at most line_tol pixels -
                            # we could save time by not doing an ellipse fit.  Currently I've set line_tol = 0
                            # which is equivalent to *not* doing this optimization.  But, in case we want to make
                            # this optimization later, I haven't deleted this code.  Simply change line_tol to
                            # line_tol = 3 pixels or whatever you like and it will happen automatically.
                            
                            line_tol = 0
                            
                            if max(lineError)>line_tol:
                                # we have to put the ellipse fit inside a try/except statement because it can fail sometimes.
                                try:
                                    ellipseEquation,ellipseError = curveFitOptimized.compute_guaranteedellipse_estimates(cluster)
                                    not_degenerate = Ellipse.not_degenerate_ellipse(ellipseEquation)
                                except:
                                    logging.debug("Ellipse Fit Failed")
                                    ellipseError = math.inf
                                    not_degenerate = False
                            else:
                                # don't bother with ellipse fit
                                ellipseError = math.inf
                                not_degenerate = False

                            # choose whether to go with the line fit or the ellipse fit.  The basic idea is to go with whichever error
                            # vector is smaller in magnitude...but doing this can sometimes cause very skinny one or two pixel wide ellipses
                            # to be fit to what should be lines.  So, I label such ellipses as "degenerate" and don't allow degenerate ellipses.
                            if np.linalg.norm(lineError)>np.linalg.norm(ellipseError) and not_degenerate:
                                isEllipse = True
                                error_vec = ellipseError
                                equation = ellipseEquation
                            else:
                                isEllipse = False
                                error_vec = lineError
                                equation = lineEquation
                            myCurve = curve.Curve(cluster,equation,error_vec,isEllipse,cols,rows)
                            self.addcurve(myCurve)
                            for ll in range(len(cluster)):
                                fragmentMap[int(cluster[ll][1]),int(cluster[ll][0])]=255
        # in debug mode, save a map of all the fragments.
        if self.debug > 0:
            debug_path = self.debugFolderName+"/fragmentMap.png"
            cv2.imwrite(debug_path,fragmentMap)
        # if we are in deep debug mode, log all the info about the skeleton.
        if self.debug > 1:
            self.printInfo();

    def addcurve(self,C):

            # C is a curve.

            # There are three factors to contemplate when
            # considering adding C to the skeleton.

            # First, we don't want lots of small curves in our skeleton.
            # We only care about the curves that are either horizontally or
            # verically complete.
            # Therefore, if the curve is neither, we don't add it.

            # Second, is it redundant with one
            # or more curves already in the skeleton?
            # If not, we add it to the skeleton.
            
            # Third, if it is redudant with one or more existing
            # curves, does it have a higher confidence value than them?
            # If it does, we delete all of those curves, and replace them
            # with this one.  Othwerise, we don't add the curve.

            if not C.horizontally_complete and not C.vertically_complete:
                return
       
            if len(self.curves)==0:
                self.curves.append(C)
            else:
                N = len(self.curves)
                redundant = False
                better_indices = []
                better_count = 0
                # search over existing curves, checking for redundance.
                for j in range(N):
                    C1 = self.curves[j]
                    if C.isRedundant(C1):
                        redundant = True
                        # ok, but maybe it's better than the curve already
                        # there?
                        if C.confidence > C1.confidence:
                            better_count = better_count+1
                            better_indices.append(j)
                if redundant and len(better_indices)>0:
                    # delete the less good curves
                    for j in range(better_count):
                        self.curves[better_indices[j]]=None
                        
                    # get rid of the "None"s.  Python has a "remove" function to do this,
                    # but it was causing me errors, so let's do it ourselves with a loop.
                    newCurves = []
                    for j in range(len(self.curves)):
                        if not self.curves[j]==None:
                            newCurves.append(self.curves[j])
                    self.curves = newCurves
                    # add in our curve
                    self.curves.append(C)

                if not redundant:
                    self.curves.append(C)
            return

    def printInfo(self):
        logging.debug("##### Printing Skeleton####")
        for j in range(len(self.curves)):
            C = self.curves[j]
            logging.debug("----Skeleton Curve number ---",j+1," of ",len(self.curves))
            C.printInfo()
        logging.debug("##### Done Printing Skeleton####")
        return

    def draw(self,filename,onlyUseful):
        # draw curves on top of the image.  Lines will be drawn in blue and ellipses in red.
        
        thickness = 5 # thickness of the curves to be drawn, in pixels.

        # we have to make a copy, or else pass by reference will be performed and self.image
        # will be modified as a result.
        image2 = copy.copy(self.image)

        if not onlyUseful:
            curvesToPlot = self.curves
        else:
            curvesToPlot = self.sortedUsefulCurves
        
        for C in curvesToPlot:
            if not C.isEllipse:
                if len(C.endPoints)==2:
                    p1 = (int(C.endPoints[0][0]),int(C.endPoints[0][1]))
                    p2 = (int(C.endPoints[1][0]),int(C.endPoints[1][1]))
                    cv2.line(image2,p1,p2,(255,0,0),thickness)
            else:
                a,b,x0,y0,phi=Ellipse.convertEllipseToGeometricForm(C.equation)
                phi = phi*180/(math.pi)
                # the above code could possibly return phi ~= 180.  Obviously an ellipse rotated
                # 180 degrees is just the ellipse back.  But this case will complicate the logic
                # below, where we are only drawing a partial arc of the ellipse.  So, we avoid
                # this by subtracting 180 from phi if we notice it taking on a larger value than 90.
                if phi>90:
                    phi = phi-180

                # the above code will set phi so that a > b always, by possibly setting phi ~= +/-90.
                # But this makes the logic of drawing the correct segment of the ellipse more complex.
                # Instead, if we notice that phi is roughly +/-90 (within 10 degrees),
                # we make roughly 0 by adding or
                # subtracting 90.  Then we swap a and b.
                if abs(phi-90)<10:
                    atemp = a
                    a = b
                    b = atemp
                    phi = phi-90
                if abs(phi+90)<10:
                    atemp = a
                    a = b
                    b = atemp
                    phi = phi+90
                    
                
                if C.concavity=='down':
                    startAngle = 0
                    endAngle = 180
                if C.concavity=='up':
                    startAngle = -180
                    endAngle = 0
                if C.concavity=='leftwards':
                    startAngle = -90
                    endAngle = 90
                if C.concavity=='rightwards':
                    startAngle = 90
                    endAngle = 270                   
                cv2.ellipse(image2,(int(x0),int(y0)),(int(a),int(b)),phi,startAngle,endAngle,(0,0,255),thickness);
        cv2.imwrite(filename,image2)
        return

    def isRotated(self):

        # The basic idea here is to compute the maximum curvature
        # of the vertically complete curves, and compare it with
        # that of the horizontally complete curves.  If it is bigger,
        # and if we also have at least one "concave leftwards"
        # or "concave rightwards" curve, we conclude that or image
        # is indeed rotated.

        # here it is the magnitude of the curvature that matters, so
        # we get rid of the sign by taking an absolute value
            
        horizontal_curvatures=[]
        vertical_curvatures=[]

        num_left = 0
        num_right =0

        for k in range(len(self.curves)):
           C = self.curves[k]
           if C.isEllipse:
               if C.horizontally_complete:
                   horizontal_curvatures.append(abs(C.curvature))
               if C.vertically_complete:
                   vertical_curvatures.append(abs(C.curvature))
                   if C.concavity=='leftwards':
                       num_left = num_left+1
                   elif C.concavity=='rightwards':
                       num_right = num_right+1

        if len(vertical_curvatures)>0:
            max_vert = max(vertical_curvatures);
        else:
            max_vert = 0
        if len(horizontal_curvatures)>0:
            max_horiz = max(horizontal_curvatures)
        else:
            max_horiz = 0
        if (num_left > 1 or num_right > 1 ) and ( max_vert > max_horiz ):
            return True
        else:
            return False

    # flip the skeleton about the line y=x.  This means we flip both all
    # curves in the skeleton, as well as self.image.
    def flip(self):
        newCurves=[];
        for j in range(len(self.curves)):
            C = self.curves[j]
            flipped_C = C.flip()
            newCurves.append(flipped_C)
        self.curves=newCurves
        # flip image about the line y=x.
        self.image = skeletonHelperFunctions.flipImageAbout45(self.image)
        return

    # this function will only be invoked after we have (if necessary)
    # flipped the skeleton about y=x.  We now go through the image, selecting
    # the curves which are the most "useful" (as explained in the accompanying PDF)
    # for the purposes of unwarping the image, and sorted them according to their y-coordinate.
    def getSortedUsefulCurves(self):
           
        usefulCurves =[]
        y_vec=[]
        num_useful_curves = 0

        for j in range(len(self.curves)):
           C = self.curves[j]
           if C.isUseful():
               num_useful_curves = num_useful_curves + 1
               x_extremal = C.x_extremal
               y_vec.append(x_extremal[1])
               usefulCurves.append(C)                  
        perm = np.argsort(y_vec)

        sortedUsefulCurves=[]
        sorted_y_vec=[]
        
        for j in range(len(y_vec)):
            sortedUsefulCurves.append(usefulCurves[perm[j]])
            sorted_y_vec.append(y_vec[perm[j]])

        self.y_vec = sorted_y_vec
        self.sortedUsefulCurves = sortedUsefulCurves
        return

    # Our sorted useful curves should have monotonically decreasing signed curvature
    # as you move from the top of the image to the bottom (however image coordinates start at the top
    # of the image and increase as you move down, so relative to image coordinates this is the other way
    # around).

    # Enforcing this constraint helps elimate false-positive curves.  In this function, the constraint is enforced
    # by seeking a "longest increasing subsequence" of curvatures.  This is a standard problem
    # (see https://en.wikipedia.org/wiki/Longest_increasing_subsequence)
    # And we solve it in the standard way.

    # The field "afterPC" stands for "after perspective correction".  In general, this function will be called twice - once
    # before and once after perspective correction, because perspective correction can affect curvatures and affect which curves
    # are deemed significant.  We only need this field if we are in debug mode and will be saving graphs of the curvature before
    # and after enforcing monotonicity - we use this field to decide how to label the graphs.

    # this function also returns the average magnitude of the curvature of the curves, after monotonicity has been
    # enforced.  This provides a useful measure of how "warped" the image is.  To make the answer independent
    # of the size of the image, we convert to a coordinate system in which the image has width 1.  This is
    # equivalent to multiplying the curvature by the width of the image.
    
    def enforceMonotonicity(self,afterPC):


        # The code here is concerned merely with drawing a graph of the curvatures, possibly displaying it,
        # and possibly saving it to disc (depending on which level of debug mode we are in).
        x=[]
        y=[]

        for k in range(len(self.sortedUsefulCurves)):
            x.append(self.y_vec[k])
            y.append(self.sortedUsefulCurves[k].curvature)
        if self.debug > 0:
            fig = plt.figure()
            plt.scatter(x,y,color='red')
            plt.ylabel('signed curvature')
            plt.xlabel('horizontal distance down image')
            if not afterPC:
                plt.title('signed curvature prior to enforcing monotonicity (PC not yet done)')
            else:
                plt.title('signed curvature prior to enforcing monotonicity (PC already done)')
            y_max = max(y)
            y_min = min(y)
            if y_max > y_min:
                plt.axis([min(x), max(x), y_min, y_max])
            if not afterPC:
                debug_path = self.debugFolderName+"/curvaturesBeforeMonotonicityPCnotYetDone.png";
            else:
                debug_path = self.debugFolderName+"/curvaturesBeforeMonotonicityPCalreadyDone.png";
            fig.savefig(debug_path)
            if self.debug > 1:
                plt.show()
            else:
                plt.close(fig)
                
        # Here is where we actually enforce the constraint.

        # First we find the indices of a "longest increasing subsequence".  I is an array of length
        # len(self.sortedUsefulCurve) such that I[j]=1 if self.sortedUsefulCurves[j] belongs to the
        # subsequence and I[j]=0 otherwise.

        I = skeletonHelperFunctions.longestSubsequence(y);

        # Now we overwrite "self.y_vec" and "self.SortedUsefulCurves" by deleting curves for which I[j]=0.
    
        new_SortedUsefulCurves = []
        new_y_vec=[]
        new_x = []
        new_y = []

        for k in range(len(self.sortedUsefulCurves)):
            if I[k]==1:
                new_SortedUsefulCurves.append(self.sortedUsefulCurves[k])
                new_y_vec.append(self.y_vec[k])
                new_x.append(x[k])
                new_y.append(y[k])

        self.sortedUsefulCurves = new_SortedUsefulCurves
        self.y_vec = new_y_vec

        # compute some statistics on the curvature, post monotonicity enforcing,
        # in a coordinate system where the image has width 1.  This is equivalent to multiplying the curvature
        # in our current coordinate system by the width of the image, which we first need to get.
        im_width= self.image.shape[1]
        curvatureAbs = np.sort(np.abs(new_y)*im_width)
        N_curv = len(curvatureAbs)-1 # what I really mean here is not the number of elements, rather the largest index in the array.
        avgCurvature = np.mean(curvatureAbs)
        varCurvature = np.var(curvatureAbs)
        minCurvature = curvatureAbs[0]
        maxCurvature = curvatureAbs[N_curv]
        curv_25percentile = 0.5*(curvatureAbs[math.floor(N_curv*0.25)]+curvatureAbs[math.ceil(N_curv*0.25)])
        curv_75percentile = 0.5*(curvatureAbs[math.floor(N_curv*0.75)]+curvatureAbs[math.ceil(N_curv*0.75)])
        curvatureStats = np.array([avgCurvature,varCurvature,minCurvature,maxCurvature,curv_25percentile,curv_75percentile])

        # The remaining code is concerned with possibly making some graphs of how things look after enforcing
        # this constraint.

        if self.debug > 0:
            fig = plt.figure()
            plt.scatter(new_x,new_y,color='red')
            plt.ylabel('signed curvature')
            plt.xlabel('horizontal distance down image')
            if not afterPC:
                plt.title('signed curvature after enforcing monotonicity (PC not yet done)')
            else:
                plt.title('signed curvature after enforcing monotonicity (PC already done)')
            y_max = max(y)
            y_min = min(y)
            if y_max > y_min:
                plt.axis([min(x), max(x), y_min, y_max])
            if not afterPC:
                debug_path = self.debugFolderName+"/curvaturesAfterMonotonicityPCnotYetDone.png"
            else:
                debug_path = self.debugFolderName+"/curvaturesAfterMonotonicityPCalreadyDone.png"
            fig.savefig(debug_path)
            if self.debug > 1:
                plt.show()
            else:
                plt.close(fig)
        return curvatureStats

    # perspective correction requires a "box" formed by two lines on the left and right
    # and a curve (either line or ellipse) on the top and bottom.  I'm currently demanding
    # that the left/right lines and top/bottom curves must lie in the left/right and top/bottom
    # of the image.  But this may not always be appropriate and we may want to change later.

    # The basic idea is to find the curve of highest confidence in each of the four regions of the image.
    # For the top and bottom curve, by doing this after monotonicity has been enforced, we have extra confidence
    # that the curve we have selected is a "good" curve.  Unfortunately, there is no monotonicity constraint for
    # the left and right lines and hence they are more vulnerable to false positives.
    
    def getLeftRightLinesTopBottomCurvesForPerspectiveCorrection(self):

        line_left,line_right=self.getLinesForPerspectiveCorrection()

        top_curve=None
        bottom_curve=None

        best_top_confidence = 0
        best_bottom_confidence = 0

        for j in range(len(self.sortedUsefulCurves)):

           C=self.sortedUsefulCurves[j]
           y=self.y_vec[j]
               
           if y<=C.im_height/2 and C.confidence > best_bottom_confidence:
               
               best_bottom_confidence=C.confidence
               bottom_curve = C
               
           elif y>C.im_height/2 and C.confidence > best_top_confidence:
               best_top_confidence=C.confidence
               top_curve = C
        return line_left,line_right,top_curve,bottom_curve

    # sub-routine of the above function that deals with the lines on the left and right specifically.
    def getLinesForPerspectiveCorrection(self):

        line_left = None
        line_right = None

        best_left_confidence = 0
        best_right_confidence = 0

        for j in range(len(self.curves)):
            C = self.curves[j]

            # we only care about lines, and they have to be vertically complete!
            if (C.isEllipse==False) and C.vertically_complete:
            
                X = C.equation
                a=X[3];b=X[4];c=X[5] # line of the form ax+by+c=0
                y_mid = math.ceil(C.im_height/2)
                # our line should be *roughly vertical* - we demand that it lie within a cone
                # making a 45 degree angle with the y-axis.
                v = np.array([-b,a]) # vector parallel to the line.
                v = v/np.linalg.norm(v)
                phase = math.atan2(v[1],v[0])

                if not a==0 and (math.pi/4<=abs(phase) and abs(phase)<=3*math.pi/4):
                    x_mid = -(c+b*y_mid)/a
                    if x_mid<=C.im_width/2 and C.confidence > best_left_confidence:
                        
                        best_left_confidence=C.confidence
                        line_left = C
                        
                    elif x_mid>C.im_width/2 and C.confidence > best_right_confidence:
                        best_right_confidence=C.confidence
                        line_right = C
        return line_left,line_right;

    # This is a routine to draw our "box" to be used for perspective correction, including
    # the four intersection points of the curves.  There is some duplication of code with
    # the standard "draw" function.  I apologize.

    def drawLinesAndTopBottomCurves(self,filename,line_left,line_right,top_curve,bottom_curve):

        thickness = 5 # thickness of the curves.

        # we have to make a copy, or else pass by reference will be performed and self.image
        # will be modified as a result.
        image2 = copy.copy(self.image)

        # this first part is exactly the same as our other "draw routine".

        curvesToPlot=[line_left,line_right,top_curve,bottom_curve]
        
        for C in curvesToPlot:
            if not C.isEllipse:
                if len(C.endPoints)==2:
                    p1 = (int(C.endPoints[0][0]),int(C.endPoints[0][1]))
                    p2 = (int(C.endPoints[1][0]),int(C.endPoints[1][1]))
                    cv2.line(image2,p1,p2,(255,0,0),thickness)
            else:
                a,b,x0,y0,phi=Ellipse.convertEllipseToGeometricForm(C.equation)
                phi = phi*180/(math.pi)
                # the above code could possibly return phi ~= 180.  Obviously an ellipse rotated
                # 180 degrees is just the ellipse back.  But this case will complicate the logic
                # below, where we are only drawing a partial arc of the ellipse.  So, we avoid
                # this by subtracting 180 from phi if we notice it taking on a larger value than 90.
                if phi>90:
                    phi = phi-180

                # the above code will set phi so that a > b always, by possibly setting phi ~= +/-90.
                # But this makes the logic of drawing the correct segment of the ellipse more complex.
                # Instead, if we notice that phi is roughly +/-90 (within 10 degrees),
                # we make roughly 0 by adding or
                # subtracting 90.  Then we swap a and b.
                if abs(phi-90)<10:
                    atemp = a
                    a = b
                    b = atemp
                    phi = phi-90
                if abs(phi+90)<10:
                    atemp = a
                    a = b
                    b = atemp
                    phi = phi+90
                    
                
                if C.concavity=='down':
                    startAngle = 0
                    endAngle = 180
                if C.concavity=='up':
                    startAngle = -180
                    endAngle = 0
                if C.concavity=='leftwards':
                    startAngle = -90
                    endAngle = 90
                if C.concavity=='rightwards':
                    startAngle = 90
                    endAngle = 270                  
                cv2.ellipse(image2,(int(x0),int(y0)),(int(a),int(b)),phi,startAngle,endAngle,(0,0,255),thickness)

        # Now we also want to draw the points of intersection.  First we have to find them.
        p1,int_found1 = top_curve.getIntersectionWithLine(line_left)
        p2,int_found2 = top_curve.getIntersectionWithLine(line_right)
        p3,int_found3 = bottom_curve.getIntersectionWithLine(line_left)
        p4,int_found4 = bottom_curve.getIntersectionWithLine(line_right)

        # these will all be drawn as circles with a radius of 2.5 times the line thickness
        radius = int(2.5*thickness)

        if int_found1:
            cv2.circle(image2,(int(p1[0]),int(p1[1])),radius,(0,255,0),thickness)
        if int_found2:
            cv2.circle(image2,(int(p2[0]),int(p2[1])),radius,(0,255,0),thickness)
        if int_found3:
            cv2.circle(image2,(int(p3[0]),int(p3[1])),radius,(0,255,0),thickness)              
        if int_found4:
            cv2.circle(image2,(int(p4[0]),int(p4[1])),radius,(0,255,0),thickness)
        cv2.imwrite(filename,image2)

        return

    # this is a routine to do perspective correction based on the "box" of curves
    # we have found with the routines above.  We have to update both our array of curves
    # as well as self.image.

    def perspectiveCorrection(self,line_left,line_right,top_curve,bottom_curve):

        # we have lines like this
        # p1                  p2
        #  \                 /
        #   \               /
        #    \             /
        #     p3         p4
        
        # or maybe like this.
        
        #    p1        p2
        #   /           \      
        #  /             \   
        # /               \     
        #p3                p4
        
        # or maybe even this
        # p1             p2
        # \              \
        #  \              \
        #   \              \
        #    \              \
        #     p3             p4
        
        # we want to make them parallel and vertical.

        # First find the four points of intersection in the drawing above.

        p1,int_found1 = top_curve.getIntersectionWithLine(line_left)
        p2,int_found2 = top_curve.getIntersectionWithLine(line_right)
        p3,int_found3 = bottom_curve.getIntersectionWithLine(line_left)
        p4,int_found4 = bottom_curve.getIntersectionWithLine(line_right)

        # we can only proceed if the four points of intersection are found
        if int_found1 and int_found2 and int_found3 and int_found4:

            newBoxHeight = np.linalg.norm(p1-p3);
            newBoxWidth  = np.linalg.norm(p1-p2);

            P1 = np.array([0,newBoxHeight])
            P2 = np.array([newBoxWidth,newBoxHeight])
            P3 = np.array([0,0])
            P4 = np.array([newBoxWidth,0])

            oldImageHeight,oldImageWidth,channels = self.image.shape

            # compute a homography mapping pi-->Pi for i=1,2,3,4.

            H,homography_found = Homography.ComputeHomographyFromPoints(p1,p2,p3,p4,P1,P2,P3,P4)

            if homography_found:

                # compute the bounding box of the old image after applying the homography.  In the unlikely event that no such bounding box
                # exists, is_bounded will be False.
            
                x_min,x_max,y_min,y_max,is_bounded = Homography.getBoundingBoxFromHomography(oldImageWidth,oldImageHeight,H)
                if is_bounded:
                    # our new image we be as large as the bounding box above.
                    newImageHeight = math.ceil(y_max-y_min)
                    newImageWidth  = math.ceil(x_max-x_min)
                    # compute the new image by applying the homography to the old image
                    # at the same time we compute a mask which will be possibly be used for inpainting
                    newImage,success,mask=Homography.applyHomographyToImageUsingBoundingBox(H,self.image,x_min,x_max,y_min,y_max,self.inpaintFlag)
                    # If inpaintFlag is 3 or 4, we now use the mask above to inpaint black areas in newImage
                    # using either Telea's algorithm or a navier-stokes based approach.
                    if self.inpaintFlag == 3:
                        self.image = cv2.inpaint(newImage,mask,3,cv2.INPAINT_TELEA)
                    elif self.inpaintFlag == 4:
                        self.image = cv2.inpaint(newImage,mask,3,cv2.INPAINT_NS)
                    else:
                        self.image = newImage
                    # now, update all the curves in the skeleton by applying the homography to the curves.
                    self.perspectiveCorrectCurves(H,x_min,y_min,newImageWidth,newImageHeight)
                    
                else:
                    success = False
            else:
                success=False
        else:
            success=False
        return success

    def saveImage(self,filename):
        cv2.imwrite(filename,self.image)
        return

    # subroutine to update the curves in the skeleton using a given homography.
    def perspectiveCorrectCurves(self,H,x_min,y_min,newWidth,newHeight):
        newCurves=[]
        for j in range(len(self.curves)):
            C = self.curves[j]
            newC = C.applyHomography_and_translation(H,-x_min,-y_min,newWidth,newHeight)
            newCurves.append(newC)
        self.curves=[]
        for j in range(len(newCurves)):
            C = newCurves[j]
            self.addcurve(C)
        return

    # debugging routine that draws all the points in all the curves making up the skeleton.
    def drawPoints(self):
        for j in range(len(self.curves)):
            C = self.curves[j]
            points = C.pixels

            plt.scatter(points[:,0], points[:,1],color='red')
        plt.ylabel('image y axis')
        plt.xlabel('image x axis')
        plt.title('pixels making up image skeleton')
        plt.show()
        return

    def alreadyStraight(self):
        for j in range(len(self.sortedUsefulCurves)):
            C = self.sortedUsefulCurves[j]
            if C.isEllipse:
                return False
        return True

    def insufficientData(self):
        ellipse_count = 0

        num_curves = len(self.sortedUsefulCurves)
    
        for k in range(num_curves):
            C = self.sortedUsefulCurves[k]
            if C.isEllipse:
                ellipse_count = ellipse_count + 1

        if ellipse_count == 0 or num_curves < 2:
            return True
        else:
            return False

    # this is the main routine that does the actual unwarping of the image.
    def unwarpImage(self):
        
        rows,cols,channels=self.image.shape

        n_useful = len(self.y_vec)

        # build delta matrix corresponding to the functions
        # delta_i(x) defined in the accompanying PDF.
        Delta = np.zeros((n_useful,cols),dtype=np.float64)
        for i in range(n_useful):
            C_i = self.sortedUsefulCurves[i];
            x_extremal_i = C_i.x_extremal
            y_extremal_i = x_extremal_i[1]
            for j in range(cols):
                Delta[i,j]=C_i.get_y(j)-y_extremal_i


        # We may need to possibly pad the new image, as explained in the accompanying PDF.

        Pad = np.max(-Delta)
        rows_new = rows
        if Pad > 0:
            Pad=math.ceil(Pad)
            rows_new = rows_new+Pad

        # The actual updating of the image is done in a static routine optimized using numba.  Numba allows no objects, so the skeleton itself
        # cannot be passed to this routine - but the delta matrix we have just computed is enough.
            
        # we need to convert y_vec to an array in order to be able to use the numba optimized version
        # of the two functions below, as numba does not support lists.
        y_vec_array = np.array(self.y_vec,dtype=np.float64)

        # As explained in the accompanying PDF, unwarping is easy for pixels (x,y) with y_i < y < y_{i+1} for some y_i in y_vec.  But for
        # y < y_0 or y_{N-1} < y, we are doing extrapolation instead of interpolation and this requires additional care.  For y < y_0, linear
        # extrapolation based on y_0 and y_1 is not necessarily a good idea.  Instead it should be down based on y_0 and y_{firstPartner}, where
        # firstPartner is found using the routine below.  Details are given in the accompanying PDF.
        [firstPartner,lastPartner]=skeletonHelperFunctions.findPartners(y_vec_array,rows_new)

        # compute the unwarped image, as well as a mask that will possibly be used for inpainting
        unwarpedImage,mask = skeletonHelperFunctions.unWarp(self.image,rows_new,y_vec_array,firstPartner,lastPartner,Delta,self.inpaintFlag)
        # if inpaintFlag is 3 or 4, we now inpaint.
        if self.inpaintFlag == 3:
            self.image = cv2.inpaint(unwarpedImage,mask,3,cv2.INPAINT_TELEA)
        elif self.inpaintFlag == 4:
            self.image = cv2.inpaint(unwarpedImage,mask,3,cv2.INPAINT_NS)
        else:
            self.image = unwarpedImage
        return


    
    
    
