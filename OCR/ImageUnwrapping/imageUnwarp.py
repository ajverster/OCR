# Main Function for image unwarping.

# Example Usage:
# exitImage,exitFlag,curvatureStats = imageUnwarp(imagePath,debug,inpaintFlag);

# imagePath should be a string equal to the path to
# the image to be processed, for example "./Images/TestImage.png"
# debug is an integer and should be set to 0,1, or 2.
# inpaintFlag is an integer and should be set to 0,1,2,3, or 4.

# debug = 0 means that the code runs normally, without providing any information
# about intermediate steps.

# debug = 1 means that a folder will be created in the current directory, named based
# on the imagePath (in our example above, it would be named TestImage_debug), which will
# be filled with images of intermediate steps in the unwarping process (in particular,
# images of the image skeleton at various stages, and the image after perspective correction
# has been applied but prior to unwarping).

# debug = 2 is a "deep debug" and should only be used if you are truly perplexed.  In addition
# to everything that debug = 1 does, debug = 2 will now cause the
# code to pause at various stages throughout the calculation process and display a graphic illustration
# of what is going on.  The code will not proceed further until you have manually closed the graph.

# The unwarping process will create empty areas in the output image.  This is essentially because we are trying to reconstruct
# an unwrapped copy of the cylinder, but this unwrapped version of the cylinder contains pixels that are not currently visible.
# inpaintFlag controls how we deal with this.

# inpaintFlag = 0 means we do nothing - these pixels are left black.
# inpaintFlag = 1 means we use periodic boundary conditions to avoid black areas.
# inpaintFlag = 2 means that we clamp to the nearest pixel in the image.
# inpaintFlag = 3 means that we inpaint using Telea's algorithm (built into OpenCV).
# inpaintFlag = 4 means that we inpaint using a Navier-Stokes based algorithm also built into OpenCV.

# exitImage is either the unwarpedImage, if the algorithm succeeded, or the original unaltered image,
# if it did not succeed or if it decided that no processing is necessary.

# exitFlag is an integer equal to -1,0,1, or 2.  exitFlag=-1 means that the code determined that the image
# did not need to be processed.  exitFlag = 0 means that the algorithm believes processing is necessary, but
# failed for some reason or felt it did not have enough information to do so accurately.
# exitFlag = 1 means that the algorithm succeeded, but that only perspective correction was needed - no unwarping.
# exitFlag = 2 means that the algorithm succeeded, unwarping was done, and perspective correction
# may or may not also have been done.

# curvatureStats gives us some statistics on the average curvature magnitude of the detected elliptic arcs in the image, after outlier removal, in a coordinate system
# in which we have rescaled the image to have width 1.  It gives us various measures of how "warped" the image was prior to unwarped.  If we for some reason
# are unable to perform this calculation, then "curvatureStats" will be set to "None"

# Our stats are specifically of the following form
# curvatureStats = np.array([avgCurvature,varCurvature,minCurvature,maxCurvature,curv_25percentile,curv_75percentile])

import cv2
import OCR.ImageUnwrapping.skeleton as skeleton
import OCR.ImageUnwrapping.Crop as Crop
import os
import logging

def imageUnwarp(imagePath=None, image=False, debug=False, inpaintFlag=False):
    assert sum([imagePath is None, image is None])

    # first of all, let's set the logging level.
    if debug:
        assert imagePath is not None

    logging.basicConfig(level=logging.INFO)

    # first load the image based on the path,
    # which will be something like
    # imagePath = "./images/TestImage1.png"
    if imagePath is not None:
        image = cv2.imread(imagePath)

    # now, we want to extract the part of the path
    # that just has the filename, without the folders, as well
    # as the image name "stub" (i.e. with the extension - .jpg, .png,
    # or whatever it may be) removed.
    
    # This is because, if we are in debug mode, we'll want
    # to create a bunch of intermediate images which will
    # be named based on this.  In our example above,
    # we will have imageName = "TestImage1.png" and
    # imageStub = "TestImage1".

    if imagePath is not None:
        imageName,imageStub = getImageNameAndStub(imagePath)
        debugFolderName = imageStub+"_debug"
    else:
        debugFolderName="debug"

    # if we are in debug mode, create a directory with the above name
    # if it does not already exist
    if debug > 0:
        try:
            os.mkdir(debugFolderName)
        except:
            # this means the directory already exists.
            pass

    # many images contain a lot of blank space around the label
    # So we first want to crop this away.

    cropped_image = Crop.crop_using_edges(image)

    # if we're in debug mode, let's save this cropped image.
    if debug > 0:
        debug_path = debugFolderName+"/cropped.png"
        cv2.imwrite(debug_path,cropped_image)

    # Our next step - which will be quite time consuming -
    # is to extract from our cropped image a skeleton of curves.
    mySkeleton = skeleton.Skeleton(cropped_image,debugFolderName,debug,inpaintFlag)

    # If no curves are found, we should make a note that
    # the algorithm has failed and exit.

    if len(mySkeleton.curves)==0:
        logging.info("skeleton is empty")
        exitFlag = 0
        exitImage = image
        return exitImage,exitFlag,None

    # Otherwise, if we are in debug mode, we want to create an image of the
    # skeleton and save it.  As explained in greater detail in the accompanying
    # PDF, the Skeleton has both "useful" curves and "not useful" curves.
    # So we first have to set a flag as to which kind we want to draw.  For now,
    # we draw everything.

    if debug > 0:
        onlyDrawUsefulCurves = False
        debug_path = debugFolderName+"/InitialSkeleton.png"
        mySkeleton.draw(debug_path,onlyDrawUsefulCurves)

    # Some of the images we might work on
    # are rotated 90 degrees from what we would like.
    # We now analyze the skeleton to determine if this is the case.
    # If it is, we flip both the image and the skeleton about the line
    # y=x (the skeleton stores a copy of the image inside it, so the
    # flipping of the image happens within the flipping of the skeleton).

    isRotated = mySkeleton.isRotated()

    if isRotated:
        mySkeleton.flip()
        # In debug mode, let's draw the skeleton after flipping.
        if debug > 0:
            onlyDrawUsefulCurves = False
            debug_path = debugFolderName+"/SkeletonAfterFlipping.png"
            mySkeleton.draw(debug_path,onlyDrawUsefulCurves)

    # Our image should now be nicely vertical.  Our next step
    # is to extract "useful" horizontal curves and sort them
    # according to their height.
    
    mySkeleton.getSortedUsefulCurves()

    # If no useful curves are found, we should make a note that
    # the algorithm has failed and exit.

    if len(mySkeleton.sortedUsefulCurves)==0:
        logging.info("skeleton not empty, but zero useful curves detected")
        exitFlag = 0
        exitImage = image
        return exitImage,exitFlag,None

    # The curves we have just found, if they truly do represent
    # circular cross sections of our cylinder, should obey a constraint.
    # Namely, their signed curvature should monotonically decrease
    # As we move from the top of the image to the bottom.  Enforcing
    # This constraint will help us eliminate "false positive" curves.

    # If we're in debug mode, we'll draw these curves before and after
    # enforcing the constraint.  Otherwise, we'll just enforce it.
    if debug > 0:
        onlyDrawUsefulCurves = True
        debug_path = debugFolderName+"/SortedCurvesBeforeMonontonicityConstraintPrePC.png"
        mySkeleton.draw(debug_path,onlyDrawUsefulCurves)
    # we have not yet done perspective correction
    PC_done = False;
    curvatureStats = mySkeleton.enforceMonotonicity(PC_done)
    if debug > 0:
        onlyDrawUsefulCurves = True
        debug_path = debugFolderName+"/SortedCurvesAfterMonontonicityConstraintPrePC.png"
        mySkeleton.draw(debug_path,onlyDrawUsefulCurves)

    # If enforcing the above constraint has dropped the number of useful curves to zero,
    # Then the algorithm has failed.

    if len(mySkeleton.sortedUsefulCurves)==0:
        logging.info("after enforcing monotonicity, the skeleton had no useful curves detected")
        exitFlag = 0
        exitImage = image
        return exitImage,exitFlag,curvatureStats

    # Our next step is to determine whether or not perspective correction is
    # Required for our image, and if so, to perform it.

    # The basic way we go about this is by looking for a "box" in our image
    # Consisting of lines on the left and right, and a top curve and bottom
    # curve which may be lines or could also be ellipses.
    # The following function finds these curves, or returns "None" if a
    # given curve cannot be found.
    
    line_left,line_right,top_curve,bottom_curve=mySkeleton.getLeftRightLinesTopBottomCurvesForPerspectiveCorrection()

    # This is a flag to tell us if perspective correction succeeded.
    # If it does succeed, this flag will later be set to True.
    PC_succeeded = False

    # we need the above lines to exist if we are to do perspective correction
    if line_left!=None and line_right!=None and top_curve!=None and bottom_curve!=None:

        # if we are in debug mode, let's draw this box.
        if debug > 0:
            debug_path = debugFolderName+"/Box.png"
            mySkeleton.drawLinesAndTopBottomCurves(debug_path,line_left,line_right,top_curve,bottom_curve)

        # perspective correction is only necessary if the lines are not already perfectly vertical
        
        if not line_left.isVerticalLine() or not line_right.isVerticalLine():

            PC_succeeded = mySkeleton.perspectiveCorrection(line_left,line_right,top_curve,bottom_curve)
            
            # if we're in debug mode, and perspective correction suceeded, let's draw the new image
            # both with and without the skeleton overlaid.
            if PC_succeeded and debug > 0:
                debug_path = debugFolderName+"/PerspectiveCorrection.png"
                mySkeleton.saveImage(debug_path)

                onlyDrawUsefulCurves = False
                debug_path = debugFolderName+"/SkeletonAfterPerspectiveCorrection.png"              
                mySkeleton.draw(debug_path,onlyDrawUsefulCurves)


    if PC_succeeded:
        logging.info("perspective correction suceeded")
        # Perspective correction may have changed which curves are the most "useful"
        # So, we select our useful curves, and enforce monotonicity a second time.
        # Once again, if the number of useful curves either pre- or post- monotonicity
        # is zero, the algorithm has failed and we exit.
        
        mySkeleton.getSortedUsefulCurves();
        if len(mySkeleton.sortedUsefulCurves)==0:
            logging.info("After perspective correction, zero useful curves detected")
            exitFlag = 0
            exitImage = image
            return exitImage,exitFlag,curvatureStats
        # we have done perspective correction
        PC_done = True

        if debug > 0:
            onlyDrawUsefulCurves = True
            debug_path = debugFolderName+"/SortedCurvesBeforeMonontonicityConstraintPostPC.png"
            mySkeleton.draw(debug_path,onlyDrawUsefulCurves)
        
        mySkeleton.enforceMonotonicity(PC_done)
        if len(mySkeleton.sortedUsefulCurves)==0:
            logging.info("After perspective correction and enforcing monotonicity, zero useful curves detected")
            exitFlag = 0
            exitImage = image
            return exitImage,exitFlag,curvatureStats

        if debug > 0:
            onlyDrawUsefulCurves = True
            debug_path = debugFolderName+"/SortedCurvesAfterMonontonicityConstraintPostPC.png"
            mySkeleton.draw(debug_path,onlyDrawUsefulCurves)

    # many images don't need to be straightened, so the first thing to do is
    # determine whether or not any straightening is necessary.
    if mySkeleton.alreadyStraight():
        if PC_succeeded: 
            logging.info('only perspective correction was needed')
            exitFlag = 1
            exitImage = mySkeleton.image
            return exitImage,exitFlag,curvatureStats
        else:
            logging.info('this image required no modification')
            exitFlag = -1
            exitImage = image
            return exitImage,exitFlag,curvatureStats
            

    # on the other hand, it might be that the image *is* in need of
    # straightening, but we lack the data necessary to do so.
    elif mySkeleton.insufficientData():
        
        logging.info('insufficient useable curves.  Image could not be straightened.')
        exitFlag = 0
        exitImage = image
        return exitImage,exitFlag,curvatureStats

    # finally, if we have determined that our image needs to be straightened,
    # and we have what we need to do it, then we go ahead and do it.  
    else:
        
        mySkeleton.unwarpImage()
        
        # don't forget that if we flipped the image before, we need to now undo
        # this
        if isRotated:
            mySkeleton.flip()
            
        logging.info('unwarping succeeded!')
        exitFlag = 2
        exitImage = mySkeleton.image
        return exitImage,exitFlag,curvatureStats

# a helper function to extract the image name and stub from a path
def getImageNameAndStub(imagePath):

    # remove the portion of the image path containing any folders.
    imageName = imagePath.split("/")[-1]

    # Now we want to remove the extension (e.g. ".png"), to get the "stub"
    # However, we have to do this in a robust way so that filenames like
    # '61rl2HVs4kL._SL1500_.jpg' - which have more than one "." in them,
    # are handled correctly.  In this case we should have
    # imageStub = '61rl2HVs4kL._SL1500_.'
    splitName = imageName.split(".")
    imageStub=""
    for k in range(len(splitName)-1):
        imageStub=imageStub+splitName[k]+"."
    imageStub = imageStub[0:len(imageStub)-1]

    return imageName,imageStub;

