import cv2
import math

def crop_using_edges(image):

    # we're going to run canny edge detection on the 
    # image, and then find the smallest and largest values
    # of i and j (i and j are row and column indices
    # respectively) where the edge map is non-zero.

    # call these values i_min,i_max,j_min,j_max.

    # Essentially, what we want is to crop the image down
    # to the rectangle [i_min,i_max] x [j_min,j_max].
    # However, it will be useful to have some
    # padding.  So what we'll actually do is crop down to
    # [i_min-pad,i_max+pad] x [j_min-pad,j_max+pad].
    # However, it may possibly be the case that i-min-pad<0
    # or i_max+pad >= rows ("rows" is the number of rows
    # in the original image).  So what we REALLY want to do
    # is crop to
    # [max(i_min-pad,0),min(i_max+pad,rows-1)]
    # x [max(j_min-pad,0),min(j_max+pad,cols-1)].

    pad = 10 # pad with 10 pixels

    # run canny edge detection with lower theshold of
    # 100 and upper of 200 (these numbers assume image values
    # run between 0 and 255, and do not make sense if this is
    # not the case)
    
    edges = cv2.Canny(image,5,200,L2gradient=True)

    # the edge image is 2D - it only has one channel

    rows,cols = edges.shape

    i_min = math.inf
    j_min = math.inf
    i_max = -math.inf
    j_max = -math.inf

    for i in range(rows):
        for j in range(cols):
            c = edges[i,j]
            if c>0:
                if i>i_max:
                    i_max = i
                if i<i_min:
                    i_min = i
                if j>j_max:
                    j_max = j
                if j<j_min:
                    j_min = j            

    i_min = max(i_min-pad,0)
    i_max = min(i_max+pad,rows-1)
    j_min = max(j_min-pad,0)
    j_max = min(j_max+pad,cols-1)
    return image[i_min:i_max,j_min:j_max]

    
