import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import math
import numba
from numba import jit

def getImageGradient(image):

    # we're going to convert the image to LAB space - our image gradient
    # will actually be the gradient of the "L" component.
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # now we're going to smooth with Gaussian blur.
    lab_image = cv2.GaussianBlur(lab_image,(5,5),0)

    # do the actual computation inside a static function optimized with numba.
    grad,meanNorm,varNorm=getGradientWithStats(lab_image)
    
    return grad,meanNorm,varNorm

@jit(nopython=True)
def getGradientWithStats(lab_image):

    rows,cols,channels=lab_image.shape

    grad = np.zeros((rows,cols,2),dtype=np.float64)

    meanNorm = 0

    for i in range(rows):
        for j in range(cols):
            # first do the x-component of the gradient
            if j==1:
                grad[i,j,0]=np.float64(lab_image[i,j+1,0])-np.float64(lab_image[i,j,0])
            elif j==cols-1:
                grad[i,j,0]=np.float64(lab_image[i,j,0])-np.float64(lab_image[i,j-1,0])
            else:
                grad[i,j,0]=(np.float64(lab_image[i,j+1,0])-np.float64(lab_image[i,j-1,0]))/2
            # now do the y-component
            if i==1:
                grad[i,j,1]=np.float64(lab_image[i+1,j,0])-np.float64(lab_image[i,j,0])
            elif i==rows-1:
                grad[i,j,1]=np.float64(lab_image[i,j,0])-np.float64(lab_image[i-1,j,0])
            else:
                grad[i,j,1]=(np.float64(lab_image[i+1,j,0])-np.float64(lab_image[i-1,j,0]))/2
            meanNorm=meanNorm+(grad[i,j,0]**2+grad[i,j,1]**2)**0.5

    meanNorm = meanNorm/(rows*cols)
    varNorm = 0
    for i in range(rows):
        for j in range(cols):
            varNorm = varNorm+((grad[i,j,0]**2+grad[i,j,1]**2)**0.5-meanNorm)**2
    varNorm = varNorm/(rows*cols)
    
    return grad,meanNorm,varNorm

# this function takes in a "fragment/component/feature" (I'm going to use these interchangeably).
# and uses clustering to split it into sub-fragments that we hope either simple curves (i.e. lines or ellipses),
# or perhaps garbage.  We will fit lines and ellipses to all the sub-fragments, the ones that are garbage will be thrown away
# and the ones that really are lines/ellipses will be kept.

def splitFragmentIntoClusters(points,image_grad,debug):
    # points is a list of lists of length 2, i.e. something of the form
    # [[x,y],[x,y],[x,y]...]
    # Let's begin by converting it to a numpy array.
    points = np.array(points)

    # Next, we want to create a numpy array consisting of the image gradient evaluated
    # at each pixel in "points"

    grads = np.zeros_like(points)
    num_pts = len(points)

    for k in range(num_pts):
        grads[k][0]=image_grad[points[k][1],points[k][0],0];
        grads[k][1]=image_grad[points[k][1],points[k][0],1];

    # if we're in deep debug mode, we plot the current fragment to be split.
    if debug>1:
        plt.scatter(points[:,0],points[:,1],color='blue')
        plt.title('Current Fragment to be Partitioned')
        plt.show()


    # we will consider from 1 up to and including 5 clusters.  However, in extreme cases, it may be that grads
    # contains fewer than 5 distict points.  We have to notice this and adapt accordingly.

    # this is accomplished by setting maxClusters to 5 by default, but reducing it if the above situation is detected.
    
    cluster_range = [1,2,3,4,5]

    maxClusters = 5;
    

    # having done this, we're now going to run kmeans clustering on the gradients, trying k=1,2,3,4,5 clusters.

    # To decide on the optimal number of clusters, we consider two metrics.  One, stored in the array "Error",
    # is simply the sum of the square roots of the variances of each cluster.  The other is the sillouette score,
    # stored in the array "S".

    # vector of sum of the square roots of the variances as a function of number of clusters.
    # it is useful to initialize this as an array of infinitys - this will simplify some of the logic
    # to come.
    Errors = (math.inf)*np.ones(5)


    # vector of sillouette values as a function of the number of clusters.
    S = np.zeros(5)

    # I am very sorry about this, but Errors[k] and S[k] are not going to tell us the Error
    # and sillouette values for k clusters - rather it will be for k+1 clusters.  This is due
    # to array indices starting at 0 in python.
    # Because of this, there are going to be strange k+1 and k-1 expressions all over the place.
    # I apologize.

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k).fit(grads)
        cluster_sizes=[]
        # first let's check that we had at least k distint points
        if max(kmeans.labels_)>=k-1:
            # the silloute score is only defined for k>=2 clusters.  It's a built in function in sklearn.metrics
            if k>1:
                S[k-1]=silhouette_score(grads,kmeans.labels_)
            # the sum of square roots of variances has to be computed manually with a loop as it isn't built in (to my knowledge).
            Errors[k-1]=0
            for kk in range(k):
                cluster_x = np.extract(kmeans.labels_==kk,grads[:,0])
                cluster_y = np.extract(kmeans.labels_==kk,grads[:,1])
                cluster_length = len(cluster_x)
                center_k = kmeans.cluster_centers_[kk]
                Errors[k-1]=Errors[k-1]+(np.sum((cluster_x-center_k[0])**2+(cluster_y-center_k[1])**2))**0.5/cluster_length**0.5
        else:
            # if we enter here, it means that k-1 clusters was the maximum allowable number of clusters.
            # we stop filling S and Errors, but because of the way we initialized them, they are already filled
            # with the correct values to make the logic below work.
            maxClusters = k-1
            break

    
    # now it is time to choose the number of clusters.  We first do this separately according to our two metrics.
    k_best_sillouette = 1;
    k_best_variances = 1;
    for k in range(1,maxClusters): 
        
        # Here's the confusing bit I alluded to above
        if S[k]>S[k_best_sillouette-1]:
            k_best_sillouette=k+1
        if Errors[k]<Errors[k_best_variances-1]:
            k_best_variances = k+1

    # The sillouette method allows us to evaluate the relative fitness of k means for k>=2,
    # but doesn't allow us to compare k=1 vs k=2.
    # The sum of square root of variances, on the other hand, does not have this restriction, but otherwise is
    # less reliable than sillouette.

    # The compromise I have settled on is the following.  If sum of square root variances says 1 cluster is
    # best, we believe it and go with 1 cluster.  Otherwise, we do whatever the sillouette method
    # tells us to do.

    if k_best_variances == 1:
        k_best = k_best_variances
    else:
        k_best = k_best_sillouette

    # If we're in deep debug mode, let's actually plot these error metrics as a function of the number of clusters.
    if debug > 1:
        plt.plot(cluster_range,Errors)
        plt.title('Sum of cluster standard deviations as a function of the number of clusters');
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of cluster standard deviations')
        plt.show();
        plt.plot(cluster_range,S);
        plt.title('Sillouette score as a function of the number of clusters');
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of cluster standard deviations')
        plt.show();
    

    # Ok, now that we know the correct number of clusters, we have to run kmeans again
    # to actually extract the clusters.

    # Although we ran the clustering on image gradients,
    # it's actually the input points we want to break into clusters.  Hence,
    # we're going to take the labelling we got from clustering the gradients and apply
    # it to the image points themselves.

    # we're also going to sort points according to their distance from the cluster center,
    # and throw away the 5% furthest away
    
    kmeans = KMeans(n_clusters=k_best).fit(grads)

    clusters=[]
    # we may possibly plot the different clusters - so we get an array of colors
    # for the different sub-fragments ready.
     
    colors=['blue','red','green','purple','black']   
    for k in range(k_best):
        # get the cluster center, as well as the x,y coordinates
        # of point the gradient vectors and pixel coordinates
        # associated with the cluster.
        center_k = kmeans.cluster_centers_[k]
        cluster_x = np.extract(kmeans.labels_==k,grads[:,0])
        cluster_y = np.extract(kmeans.labels_==k,grads[:,1])
        pixels_x = np.extract(kmeans.labels_==k,points[:,0])
        pixels_y = np.extract(kmeans.labels_==k,points[:,1])

        # cluster size before and after outlier removeal.
        cluster_size = len(cluster_x)
        culled_size = math.ceil(0.95*cluster_size);       
        distances = (cluster_x-center_k[0])**2+(cluster_y-center_k[1])**2

        perm = np.argsort(distances)

        cluster_x_culled=np.zeros((culled_size,1))
        cluster_y_culled=np.zeros((culled_size,1))
        pixels_culled=np.zeros((culled_size,2))

        for i in range(culled_size):
            cluster_x_culled[i]=cluster_x[perm[i]]
            cluster_y_culled[i]=cluster_y[perm[i]]
            pixels_culled[i][0]=pixels_x[perm[i]]
            pixels_culled[i][1]=pixels_y[perm[i]]

        cluster = np.array([cluster_x_culled,cluster_y_culled]).transpose()

        # plot the clusters if we are in deep debug mode
        if debug > 1:
            plt.scatter(pixels_culled[:,0],pixels_culled[:,1],color=colors[k]);
        clusters.append(pixels_culled);

    if debug > 1:
        plt.title('Proposed Partition based on '+str(k_best)+' clusters.')
        plt.show()

    return clusters

# helper function to get the bounding box of and number of pixels
# in a cluster.
def getBoundingBoxAndSize(cluster):
    # cluster is N x 2

    rows,cols = cluster.shape
    clusterSize = rows
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    for j in range(clusterSize):
        if cluster[j][0]<x_min:
            x_min = cluster[j][0]
        if cluster[j][1]<y_min:
            y_min = cluster[j][1]
        if cluster[j][0]>x_max:
            x_max = cluster[j][0]
        if cluster[j][1]>y_max:
            y_max = cluster[j][1]
    box = np.array([x_min,y_min,x_max-x_min,y_max-y_min])

    return box,clusterSize

# numbda optimized routine to flip an image about y=x.
@jit(nopython=True)
def flipImageAbout45(image):
    rows,cols,channels=image.shape
    flippedImage = np.zeros((cols,rows,channels),np.uint8)
    for i in range(cols):
        for j in range(rows):
            for k in range(channels):
                flippedImage[i,j,k]=image[j,i,k]
    return flippedImage

# this is a standard implementation of the standard O(N^2) Dynamic Programming
# method for fixing the longest non-decreasing subsequence of a sequence x.
# x=(x_0,x_1...,x_N) is a sequence.  The method returns a vector I of length N
# such that I[i]=1 if x_i is part of the longest non-decreasing subsequence and
# I[i]=0 otherwise.  See https://en.wikipedia.org/wiki/Longest_increasing_subsequence.

def longestSubsequence(x):

    # x is our input column vector
    # I is a vector of 0s and 1s.  The ones tell us indices of our longest
    # subsequence

    N = len(x)

    A = np.zeros((N,N))

    A[0,0]=1

    for j in range(1,N):  
        bestLength = 0
        for i in range(j): # up to but not including j
            if ( sum(A[:,i])>bestLength - 1 ) and ( x[j] <= min(np.extract(A[:,i]==1,x) )):
                A[:,j]=A[:,i]
                A[j,j]=1
                bestLength = sum(A[:,j])
        # A[:,j] is either empty, if we couldn't extend, or represents the best
        # extended length, if we could.  A[:,j-1] is the best length
        # obtained by NOT extending.  Either way, comparing the size of
        # A[:,j] with A[:,j-1] tells us what we need to know
        if sum(A[:,j])<sum(A[:,j-1]):
            A[:,j]=A[:,j-1]


    I = A[:,N-1]
    return I

@jit(nopython=True)
def findPartners(y_vec,imageHeight):

    # y_vec=[y_0,y_1,...y_{N-1}] is a vector of y coordinates in the image,
    # which we will use for bilinear interpolation.
    # when there exist y_i < y <= y_{i+1}, this is no problem.
    # but when y < y_0 or y_{N-1} < y, we're doing extrapolation instead of
    # interpolation.  We have to be a bit careful here because
    # extrapolation is generally more dangerous than interpolation.

    # for y < y_0, the most obvious thing would be to extrapolate based on 
    # y_0 and y_1.  However, if y_1-y_0 is very small in comparison with y_0,
    # this is not very well conditioned numerically.  It's better to use
    # y_0 and y_{firstParter}, where firstPartner is chosen in such a way
    # that y_{firstPartner}-y0 ~= y0.

    # a similar argument holds for y>y_{N-1}

    N = len(y_vec)

    firstPartner = -1

    # it may be that the gap between the y=0 and y_1 is small compared to
    # y_1-y_0.  In this case y_1 is the answer.
    if y_vec[1]-y_vec[0]>=y_vec[0]:
        firstPartner = 1
    else:
        # if this is not the case, lets see if we can find an i such that
        # y_{i+1}-y_1 >= y_0 >= y_i-y_1
        for i in range(1,N-1):
            if y_vec[i+1]-y_vec[0]>=y_vec[0] and y_vec[0]>=y_vec[i]-y_vec[0]:
                # pick the closer of the two
                if abs(abs(y_vec[i+1]-y_vec[0])-y_vec[0])<abs(abs(y_vec[i]-y_vec[0])-y_vec[0]):
                    firstPartner = i+1
                else:
                    firstPartner = i

        # if we've gotten to hear and firstPartner isn't set, then it means
        # y0 > y_{N-1} -y_0.  In this case just use y_{N-1}
        if firstPartner==-1:
            firstPartner = N-1

    lastPartner = -1

    # it may be that the gap between the y=imageHeight and y_{N-1} is small compared to
    # y_{N-1}-y_{N-2}.  In this case y_{N-2} is the answer.
    if y_vec[N-1]-y_vec[N-2]>=imageHeight-y_vec[N-1]:
        lastPartner = N-2
    else:
        # if this is not the case, lets see if we can find an i such that
        # y_{N-1}-y_{i-1} >= imageHeight-y_{N-1} >= y_{N-1}-y_i
        for i in range(2,N):
            if y_vec[N-1]-y_vec[i-1]>=imageHeight-y_vec[N-1] and imageHeight-y_vec[N-1]>=y_vec[N-1]-y_vec[i]:
                # pick the closer of the two
                if abs(abs(y_vec[N-1]-y_vec[i])-(imageHeight-y_vec[N-1]))<abs(abs(y_vec[N-1]-y_vec[i-1])-(imageHeight-y_vec[N-1])):
                    lastPartner = i
                else:
                    lastPartner = i-1

        # if we've gotten to here and lastPartner isn't set, then it means
        # imageHeight-y_{N-1} > y_{N-1}-y_0  In this case just use y_0
        if lastPartner==-1:
            lastPartner = 0
    return firstPartner,lastPartner

@jit(nopython=True)
def unWarp(image,new_height,y_vec,firstPartner,lastPartner,Delta,inpaintFlag):

    height,width,channels=image.shape

    new_image = np.zeros((new_height,width,channels),np.uint8)
    mask = np.zeros((new_height,width),np.uint8)

    n_useful = len(y_vec)

    for y in range(new_height):
        for x in range(width):
            # first, find the relevant i
            if y<=y_vec[0]:

                i = 0
                ip = firstPartner

            elif y>=y_vec[n_useful-1]:

                i = lastPartner
                ip = n_useful-1

            else:
                for ii in range(n_useful-1):
                    if y_vec[ii]<=y and y<y_vec[ii+1]:
                        i = ii
                ip = i+1


            s = (y-y_vec[i])/(y_vec[ip]-y_vec[i])
            DeltaY = (1-s)*Delta[i,x]+s*Delta[ip,x]
            yp = y+DeltaY
            Y = math.floor(yp)
            t = yp-Y

            # if inpaintFlag == 0, then no inpainting is done and no special boundary conditions are applied.
            
            if inpaintFlag == 1: # periodic boundary conditions
                Y = Y%(height-2)
            elif inpaintFlag == 2: # clamping
                if Y<0:
                    Y = 0
                    t = 0
                if Y>height-2:
                    Y = height-2;
                    t = 1;

            # if inpaintFlag == 3 or 4, we're going to inpaint.  We'll build a mask for that purpose shortly.
            
            if Y<0 or Y > height-2:
                if inpaintFlag == 3 or inpaintFlag == 4: # we only have to construct the mask if we are doing inpainting
                    mask[y,x]=1
            else:
                #t = yp-Y
                for k in range(channels):
                    new_image[y,x,k]=(1-t)*image[Y,x,k]+t*image[Y+1,x,k]
    return new_image,mask
    
