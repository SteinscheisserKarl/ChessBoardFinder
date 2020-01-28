#!/usr/bin/python

'''
Created on 2020-01-24

@author: Thomas Klaube
'''

import cv2
import numpy as np
import argparse
# from skimage.feature import peak_local_max
import scipy.ndimage as ndimage 

# will remove (set to 0) every point around a local max value in img while retaining the max value itself
# optional distance defines the region around the maximum
# this function should usually not operate on real images but on matrices containing derived values (Saddlepoints, local extrema...)
def ultimateErosion(img,distance=11):
   image_max = ndimage.maximum_filter(img, size=distance, mode='constant')   # probably this could be done with a dilation but I was to stupid...
   # coordinates = peak_local_max(img, min_distance=distance)                # coordinates of peaks points - but we dont really need this - needs import from skimage.feature...
   maxima = cv2.compare(img, image_max, cv2.CMP_EQ)                          # the region of (distance x distance) around the maxima is now 0. But the maxima itself is retained 
   img[maxima == 0] = 0                                                      # removing everything around the maxima

# just for debugging... Saddlepoints matrices contain values from -1000000 to + 1000000. Values must
# be normalized and can then be displayed as an image 
def showSaddle(img,title):
    foo = (img - img.min())/(img.max()-img.min())*255 
    abs_S = cv2.convertScaleAbs(foo)
    cv2.imshow(title,abs_S)
    cv2.waitKey(0)

# returns polar coordinates of points in arr with ref being the point of origin
def polars(arr,ref):
   x,y = arr[:,0]-ref[0], arr[:,1]-ref[1]
   r = np.sqrt(x**2+y**2)
   t = np.arctan2(y,x)
   return (r,t)

def getColor(img,length,xcoord,ycoord):    # xcoord,ycoord of "0,0" means square A1, "7,7" is H8 and 3,0 is D1
   # print (length,xcoord,ycoord)
   sl = length//8 # squarelength
   color = cv2.norm(img[(length-(ycoord+1)*sl):(length - ycoord*sl),xcoord*sl:(xcoord+1)*sl],cv2.NORM_L1)
   return color

def sameColorClass(a,b):
    if abs(a-b)<max(a,b)/10:
       return True
    return False

def findWarpPoints(file,isfile,startframe,debug):

    if isfile==True:
      image=cv2.imread(file,1)
    else: 
      cap = cv2.VideoCapture(file)
      for counter in range(startframe):
        ret, image = cap.read() 

    cap.release() 

    # searching for the chessboard is much faster on smaller images...
    # orig_rows,orig_cols,channels = image.shape        
    # if orig_cols > 600:                        # resize to 600 rows max
    #    scale = orig_cols / 600 
    #    image=cv2.resize(image,(int(orig_cols/scale),int(orig_rows/scale)))

    if debug:
       cv2.imshow('orig',image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    ddepth = cv2.CV_64F
    rows,cols = blur.shape

    grad_xx = cv2.Sobel(gray, ddepth, 2,0 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # 2nd deviation in x dir
    grad_yy = cv2.Sobel(gray, ddepth, 0,2 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # 2nd deviation in y dir

    grad_xy = cv2.Sobel(gray, ddepth, 1,1 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # deviation in x and y dir

    S = grad_xx*grad_yy - grad_xy*grad_xy   # Hesse-Matrix - holds Saddle Points (and other stuff).

    S = -1 * S   # for "technical" reasons we will negate all entries of the Hesse-Matrix as some operations
                 # on the matrix are easier to implement when searching for certain values (e.g. ultimateErosion) 

    # result of Hesse Matrix are Saddle Points (here: S). But there are _a lot_ of Saddle-Points and 
    # ohter points as well. We are interested in the negative local minima of S (as we inverted the Matrix
    # we have to look for positive local maxima)
    # everything <0 is of no interest...
    S[S<0] = 0

    S[S<0.1*S.max()]=0    # everything below 10% of the max is certainly not what we are looking for...
    ultimateErosion(S)    # now all the saddlepoints are represented as exactly one isolated value in the matrix. All values surrounding this point are 0
                          # there should be ~ 100-150 Saddlepoints in S

    # showSaddle(S,"Sattel")

    Sidx = np.where (S)
    ordering = np.argsort(S[Sidx])[::-1]
    res = np.asarray(tuple(np.array(Sidx)[:, ordering])).T  # contains coordinates of all SaddlePoints from S, ordered desc ("strongest" SaddlePoint is at res[0][0],res[1][0])

    dist = np.zeros((res.shape[0],res.shape[0]))
    for i,initpoint in enumerate(res): 
      for j,point in enumerate(res):
        dist[i,j] = np.linalg.norm(initpoint - point)       # dist[x,y] now holds the distance between points x and y of res

    len=min(rows,cols)

    orig_gray = gray.copy()
    ws = int(len/80)     # 10 percent of one squarelength
    maxoutersum=0    
    besttransform=0      # besttransform will hold the Transform Matrix for warping
    bestwarppoints=()    # warppoints
    bestwarped = 0
    xmin = cols*0.33     # The center of the chessboard must be in the region xmin:xmax, ymin:ymax
    xmax = cols*0.66   
    ymin = rows*0.33
    ymax = rows*0.66
    dst_tl=(len/8*3,len/8*3)  # these are the dimensions for the warpPerspective "target" square (corner points of the 4 inner chessboard squares)
    dst_tr=(len/8*3,len/8*5)
    dst_br=(len/8*5,len/8*5)
    dst_bl=(len/8*5,len/8*3)

    mincoord=int(len/8)
    maxcoord=int(len-len/8)

    # loop over all saddle points
    for guess,_ in enumerate(dist[0]):
      # print (guess)
      sorteddist = np.argsort(dist[guess])   # holds a sorted list of all neighbors of guess, closest neighbors come first, sorteddist[0] is guess itself 
      if not (res[sorteddist[0]][0] > ymin and res[sorteddist[0]][0] < ymax and res[sorteddist[0]][1] > xmin and res[sorteddist[0]][1] < xmax):    # outside of region - this saves a lot of time
        continue

      uips = res[sorteddist[5:9]]            # sorteddist[0-9] usually make up a 3x3 grid. The outer points of this grid (sorteddist[5:9]) span a 
                                             # quadrilateral (close to trapezoid or rectangle). I assume (guess) that this could be the quadrilateral
                                             # that surrounds the center chessboardsquares D4,E4,D5,E5
                                             # uips is an array of UnorderedInterestingPointS...

      rho,theta = polars(uips,res[sorteddist[0]])  # the uips must be ordered clockwise (or counterclockwise). Transforming to polar coordinates makes this easy
      sortedips = uips[theta.argsort()[::1]]      # the actual sorting of the uips using the sortorder of the polar thetas
      sortedips[:, 0], sortedips[:, 1] =  sortedips[:, 1], sortedips[:, 0].copy()    # switching x/y values...
      M = cv2.getPerspectiveTransform(np.float32(sortedips),np.float32([dst_tl,dst_tr,dst_br,dst_bl]))          # with the Matrix M we can now transform/warp the image 
      warped_gray = cv2.warpPerspective(orig_gray,M,(len,len),borderMode=cv2.BORDER_CONSTANT,borderValue=255)   # if the guess was correct warped_gray no holds the perfeclty warped chessboard, and nothing else...


      # now we have to verify if we really found the perfectly matching chessboard...
      grad_x = cv2.Sobel(warped_gray, ddepth, 1,0 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)   # first deviation in x
      grad_y = cv2.Sobel(warped_gray, ddepth, 0,1 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)   # first deviation in y

      #abs_grad_x = cv2.convertScaleAbs(grad_x)
      #abs_grad_y = cv2.convertScaleAbs(grad_y)

      #cv2.imshow('og',orig_gray)
      #cv2.imshow('sx',abs_grad_x)
      #cv2.imshow('sy',abs_grad_y)
      #cv2.waitKey(0)

      outersum=0.0
      for i in range(1,8): 

        os1a = cv2.norm(grad_x[0:mincoord,(i+1)*mincoord-ws:(i+1)*mincoord+ws],cv2.NORM_L1)      # only if the chessboard is matched correctly we
        os1b = cv2.norm(grad_x[maxcoord:len,(i+1)*mincoord-ws:(i+1)*mincoord+ws],cv2.NORM_L1)    # find characteristic vertical and horizontal lines 

        os2a = cv2.norm(grad_y[(i+1)*mincoord-ws:(i+1)*mincoord+ws,0:mincoord],cv2.NORM_L1)      # between all the outer squares. These can be identified by 
        os2b = cv2.norm(grad_y[(i+1)*mincoord-ws:(i+1)*mincoord+ws,maxcoord:len],cv2.NORM_L1)    # x and y deviations (Sobel)

        outersum += os1a+os1b+os2a+os2b   # the max sum hopefully represents the perfectly matched chessboard
      if outersum>maxoutersum:
         maxoutersum=outersum
         besttransform=M
         bestwarppoints=sortedips

      # cv2.waitKey(0)

    # Reverse transform
    M_inv = np.linalg.pinv(besttransform)   # inverting the Matrix M
    p_array = np.array([[[len,0]],[[0,0]],[[0,len]],[[len,len]]], dtype=np.float32)
    OuterWarpPoints = np.squeeze(cv2.perspectiveTransform(p_array, M_inv), axis=1)
    # OuterWarpPoints[:, 0], OuterWarpPoints[:, 1] = OuterWarpPoints[:, 1], OuterWarpPoints[:, 0].copy() # switching x/y vals
    bestwarped = cv2.warpPerspective(image.copy(),besttransform,(len,len))
    print ("bestwarppoints: [[%d,%d],[%d,%d],[%d,%d],[%d,%d]]" % (OuterWarpPoints[0][0],OuterWarpPoints[0][1],OuterWarpPoints[1][0],OuterWarpPoints[1][1],OuterWarpPoints[2][0],OuterWarpPoints[2][1],OuterWarpPoints[3][0],OuterWarpPoints[3][1]))

    ColE4 = getColor(bestwarped,len,4,3)  # should be white with no piece on it - very bright
    ColE5 = getColor(bestwarped,len,4,4)  # should be black with no piece on it - dark
    ColD1 = getColor(bestwarped,len,3,0)  # white with white queen              - bright
    ColD8 = getColor(bestwarped,len,3,7)  # black with black queen              - very dark
    ColA4 = getColor(bestwarped,len,0,3)  # empty white                         - very bright
    ColH4 = getColor(bestwarped,len,7,3)  # empty black                         - dark
    #print (ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)
    #print(sameColorClass(ColE4,max(ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)))
    #print(sameColorClass(ColE5,max(ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)))
    if ColE4 > ColE5:    #either the board is correctly rotated or it must be turned by 180 deg
      if ColD1 > ColD8: 
         print ("Board is correctly rotated")
      else:
         print ("rotate by 180")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_180)
    else: #the board must be rotated by 90 or 270 deg
      if ColH4 > ColA4:
         print ("Board must be rotated by 90 deg clockwise")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_90_CLOCKWISE)
      else:
         print ("Board must be rotated by 270 deg clockwise")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if debug:
      cv2.imshow("Warped",bestwarped)
      cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract moves from a Chess vid')

    parser.add_argument('file',
                        type=str,
                        help="find chessboard in this Video")

    parser.add_argument('--image',
                        action="store_true",
                        help="provided file is an image, not a video")

    parser.add_argument('--debug',
                        action="store_true",
                        help="show some debug images (will wait for keypress)")

    parser.add_argument('--startframe',
                        default="5",
                        type=int,
                        help="specify startframe in Video. Default is 5 as the first frames are often broken in videos")

    args = parser.parse_args()
    
    videoWarpPoints = findWarpPoints(args.file,args.image,args.startframe,args.debug)
    
