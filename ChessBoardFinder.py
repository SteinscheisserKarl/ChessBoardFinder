#!/usr/bin/python

'''
Created on 2020-01-24

@author: Thomas Klaube
'''
import cv2
import numpy as np
import argparse
import math
import timeit
# from skimage.feature import peak_local_max
import scipy.ndimage as ndimage 
import Undistort

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

def getCoarseMatchMask(length,ws):
    # CoarseMatchMask is used to find the derivatives around the perimeter of the chessboard which
    # are directed towards the center of the board. These derivatives can be found between any white and
    # black square and should perfectly fall into this mask if the board is precisely found... 
    # I am sure, this mask could be created much more clever... but I failed to find a better way...
    CoarseMatchMaskX = np.zeros([length,length], dtype="uint8")
    CoarseMatchMaskY = np.zeros([length,length], dtype="uint8")
    a = np.array([],np.int16)
    b = np.array([],np.int16)
    for i in range(ws+1,length-ws-1):
      if i % (length//8) <= ws:
         b = np.append(b,i)
      if i % (length//8) >= length//8 - ws:
         b = np.append(b,i)
    CoarseMatchMaskY[b, 0:length//8 ] = 255
    CoarseMatchMaskX[0:length//8, b] = 255
    CoarseMatchMaskY[b, length-length//8:length] = 255
    CoarseMatchMaskX[length-length//8:length, b] = 255
    return CoarseMatchMaskX,CoarseMatchMaskY

def findWarpPoints(image,speedup,offset,debug):

    if debug:
       cv2.imshow('orig',image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   
    ddepth = cv2.CV_64F
    rows,cols = gray.shape

    grad_xx = cv2.Sobel(gray, ddepth, 2,0 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # 2nd deviation in x dir
    grad_yy = cv2.Sobel(gray, ddepth, 0,2 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # 2nd deviation in y dir

    grad_xy = cv2.Sobel(gray, ddepth, 1,1 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)  # deviation in x and y dir

    S = grad_xx*grad_yy - grad_xy*grad_xy   # Hesse-Matrix - holds Saddle Points (and other stuff).

    S = -1 * S   # for "technical" reasons we will negate all entries of the Hesse-Matrix as some operations
                 # on the matrix are easier to implement when searching for certain values (e.g. ultimateErosion) 

    # result of Hesse Matrix are Saddle Points (here: S). But there are _a lot_ of Saddle-Points and 
    # other points as well. We are interested in the negative local minima of S (as we inverted the Matrix
    # we have to look for positive local maxima)
    # everything <0 is of no interest...
    S[S<0] = 0

    S[S<0.1*S.max()]=0    # everything below 10% of the max is certainly not what we are looking for...
    ultimateErosion(S)    # now all the saddlepoints are represented as exactly one isolated value in the matrix. All values surrounding this point are 0
                          # there should be ~ 100-300 Saddlepoints in S

    # showSaddle(S,"Sattel")

    Sidx = np.where (S)
    ordering = np.argsort(S[Sidx])[::-1]
    res = np.asarray(tuple(np.array(Sidx)[:, ordering])).T  # contains coordinates of all SaddlePoints from S, ordered desc ("strongest" SaddlePoint is at res[0][0],res[1][0], "strong" meaning steep ascents "around" the saddlepoint)
    
    if res.size > 42:           # magic number - I know... we need at least 42 saddle points - this is just my observation...
      speedup = max(0,min(100,speedup))     #  speedup must be between 0 and 100 as this is a percentage...
      res = res[0: res.size - speedup * (res.size - 42) // 100]
      # res = res[0:42]         # if speedup is 100 we will only use the "strongest" 42 saddlepoints

    dist = np.zeros((res.shape[0],res.shape[0]))
    for i,initpoint in enumerate(res): 
      for j,point in enumerate(res):
        dist[i,j] = np.linalg.norm(initpoint - point)       # dist[x,y] now holds the distance between points x and y of res

    length=min(rows,cols)

    orig_gray = gray.copy()
    ws = int(length/80)     # 10 percent of one squarelength
    MaxCoarseMatchSum=0    
    besttransform=0      # besttransform will hold the Transform Matrix for warping
    bestwarppoints=()    # warppoints
    bestwarped = 0
    xmin = cols*0.33     # The center of the chessboard must be in the region xmin:xmax, ymin:ymax
    xmax = cols*0.66   
    ymin = rows*0.33
    ymax = rows*0.66
    dst_tl=(length/8*3-1,length/8*3-1)  # these are the dimensions for the warpPerspective "target" square (corner points of the 4 inner chessboard squares)
    dst_tr=(length/8*5-1,length/8*3-1)
    dst_br=(length/8*5-1,length/8*5-1)
    dst_bl=(length/8*3-1,length/8*5-1)

    CoarseMatchMaskX,CoarseMatchMaskY = getCoarseMatchMask(length,ws)

    #cv2.imshow("CoarseMatchMaskX",CoarseMatchMaskX) # just display the mask for debugging....
    #cv2.imshow("CoarseMatchMaskY",CoarseMatchMaskY) 

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

      rho,theta = polars(uips,res[sorteddist[0]])  # the uips must be ordered clockwise. Transforming to polar coordinates makes this easy
      rhotest,thetatest = polars(res,res[sorteddist[0]])
      sortedips = np.roll(uips[theta.argsort()[::-1]],1,axis=0)      # the actual sorting of the uips using inverse sortorder of the polar thetas, cyclic shifted by one point as polars yield top right point first, but we need top left first...
      sortedips[:, 0], sortedips[:, 1] =  sortedips[:, 1].copy(), sortedips[:, 0].copy()    # switching x/y values...
      src_tl = (res[sorteddist[0]][1] - 4 * (abs(res[sorteddist[0]][1] - sortedips[0][0])),res[sorteddist[0]][0] - 4 * (abs(res[sorteddist[0]][0] - sortedips[0][1])))  # x and y in res[sorteddist] is switched!
      src_tr = (res[sorteddist[0]][1] + 4 * (abs(res[sorteddist[0]][1] - sortedips[1][0])),res[sorteddist[0]][0] - 4 * (abs(res[sorteddist[0]][0] - sortedips[1][1])))
      src_br = (res[sorteddist[0]][1] + 4 * (abs(res[sorteddist[0]][1] - sortedips[2][0])),res[sorteddist[0]][0] + 4 * (abs(res[sorteddist[0]][0] - sortedips[2][1])))
      src_bl = (res[sorteddist[0]][1] - 4 * (abs(res[sorteddist[0]][1] - sortedips[3][0])),res[sorteddist[0]][0] + 4 * (abs(res[sorteddist[0]][0] - sortedips[3][1])))
      M = cv2.getPerspectiveTransform(np.float32([src_tl,src_tr,src_br,src_bl]),np.float32([(0,0),(length-1,0),(length-1,length-1),(0,length-1)]))
      warped_gray = cv2.warpPerspective(orig_gray,M,(length,length),borderMode=cv2.BORDER_CONSTANT,borderValue=255)   # if the guess was correct warped_gray no holds the perfeclty warped chessboard, and nothing else...

      # now we have to verify if we really found the perfectly matching chessboard...
      grad_x = cv2.Sobel(warped_gray, ddepth, 1,0 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)   # first derivative in x
      grad_y = cv2.Sobel(warped_gray, ddepth, 0,1 , ksize=1, scale=5, delta=0, borderType=cv2.BORDER_DEFAULT)   # first derivative in y

      #abs_grad_x = cv2.convertScaleAbs(grad_x)
      #abs_grad_y = cv2.convertScaleAbs(grad_y)

      maskedx = cv2.bitwise_and(grad_x, grad_x, mask=CoarseMatchMaskX)
      maskedy = cv2.bitwise_and(grad_y, grad_y, mask=CoarseMatchMaskY)

      CoarseMatchSum = cv2.norm(maskedx, cv2.NORM_L1) + cv2.norm(maskedy, cv2.NORM_L1)    # if we really have a perfect fitted chessboard, then CoarseMatchSum is max for all uips....
      #print ("CMS",CoarseMatchSum)

      #cv2.imshow('absx',abs_grad_x)
      #cv2.imshow('abxy',abs_grad_y)
      #cv2.imshow('gray',warped_gray)
      #cv2.waitKey(0)

      if CoarseMatchSum>MaxCoarseMatchSum:
         MaxCoarseMatchSum=CoarseMatchSum
         besttransform=M
         bestwarppoints=np.float32([src_tl,src_tr,src_br,src_bl])

      # cv2.waitKey(0)

    OffsetTransform = cv2.getPerspectiveTransform(np.float32(bestwarppoints),np.float32([(0,0),(length-1,0),(length-1,length-1),(0,length-1)])+offset*1.0)
    M_inv = np.linalg.pinv(besttransform)   # inverting the Matrix M
    OffsetM_inv = np.linalg.pinv(OffsetTransform)   # inverting 
    p_array = np.array([[[length,0]],[[0,0]],[[0,length]],[[length,length]]], dtype=np.float32)
    offsetp_array =  np.array([[[0,0]],[[length-1+2*offset,0]],[[length-1+2*offset,length-1+2*offset]],[[0,length-1+2*offset]]], dtype=np.float32)
    OuterWarpPoints = np.squeeze(cv2.perspectiveTransform(p_array, M_inv), axis=1)
    OffsetOuterWarpPoints = np.squeeze(cv2.perspectiveTransform(offsetp_array, OffsetM_inv), axis=1)
    bestwarped = cv2.warpPerspective(image.copy(),besttransform,(length,length))
    Offsetbestwarped = cv2.warpPerspective(image.copy(),OffsetTransform,(length+2*offset,length+2*offset))
    # print ("bestwarppoints: [[%d,%d],[%d,%d],[%d,%d],[%d,%d]]" % (OuterWarpPoints[0][0],OuterWarpPoints[0][1],OuterWarpPoints[1][0],OuterWarpPoints[1][1],OuterWarpPoints[2][0],OuterWarpPoints[2][1],OuterWarpPoints[3][0],OuterWarpPoints[3][1]))

    # if correctly rotated, we could expect some of the following color-class values
    # but if incorrectly rotated some of these squares will be in different color-classes
    ColE4 = getColor(bestwarped,length,4,3)  # should be white with no piece on it - very bright
    ColE5 = getColor(bestwarped,length,4,4)  # should be black with no piece on it - dark
    ColD1 = getColor(bestwarped,length,3,0)  # white with white queen              - bright
    ColD8 = getColor(bestwarped,length,3,7)  # black with black queen              - very dark
    ColA5 = getColor(bestwarped,length,0,4)  # empty black                         - dark
    ColH4 = getColor(bestwarped,length,7,3)  # empty black                         - dark
    #print (ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)
    #print(sameColorClass(ColE4,max(ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)))
    #print(sameColorClass(ColE5,max(ColE4,ColE5,ColD1,ColD8,ColA4,ColH4)))
    rotateAngle = -1 # do not rotate
    if ColE4 > ColE5:    #either the board is correctly rotated or it must be turned by 180 deg
      if ColD1 > ColD8: 
         if debug: 
           print ("Board is correctly rotated")
      else:
         if debug:
            print ("rotate by 180")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_180)
         Offsetbestwarped = cv2.rotate(Offsetbestwarped,cv2.ROTATE_180)
         rotateAngle = cv2.ROTATE_180
    else: #the board must be rotated by 90 or 270 deg
      if ColH4 > ColA5:
         if debug:
             print ("Board must be rotated by 90 deg clockwise")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_90_CLOCKWISE)
         Offsetbestwarped = cv2.rotate(Offsetbestwarped,cv2.ROTATE_90_CLOCKWISE)
         rotateAngle = cv2.ROTATE_90_CLOCKWISE
      else:
         if debug:
             print ("Board must be rotated by 270 deg clockwise")
         bestwarped = cv2.rotate(bestwarped,cv2.ROTATE_90_COUNTERCLOCKWISE)
         Offsetbestwarped = cv2.rotate(Offsetbestwarped,cv2.ROTATE_90_COUNTERCLOCKWISE)
         rotateAngle = cv2.ROTATE_90_COUNTERCLOCKWISE
    if debug:
      cv2.imshow("Warped",bestwarped)
      cv2.imshow("Offset",Offsetbestwarped)
      cv2.waitKey(0)
    #return OuterWarpPoints,besttransform,rotateAngle
    return OffsetOuterWarpPoints,OffsetTransform,rotateAngle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract moves from a Chess vid')

    parser.add_argument('file',
                        type=str,
                        help="find the best warp points (suitable as input for warpPerspective) of the chessboard found in the first frame \
                        of this video (to use another frame see --startframe parameter)")

    parser.add_argument('--image',
                        action="store_true",
                        help="provided file is an image, not a video")
    
    parser.add_argument('--padding',
                        default="0",
                        type=int,
                        help="with padding>0 the warped and centered chessboard will be surrounded by the amount of pixels provided py PADDING")

    parser.add_argument('--noundistort',
                        action="store_true",
                        help="no undistortion of any barrel, pincushion or tangetial distortion will be performed. Use this option, if you don't have \
                        undistortion parameters (CameraMatrix, DistCoefficients) for your camera!")

    parser.add_argument('--speedup',
                        default="100",
                        type=int,
                        help="will speed up the calculation by ignoring some saddlepoints. Default is 100 (max). If you don't have undistortion parameters \
                        (see --noundistort option), you should reduce speedup to 0 as the algorithm will probably not correctly identify the chessboard corners \
                        if the barrel distorsion is significant")

    parser.add_argument('--debug',
                        action="store_true",
                        help="show some debug images (will wait for keypress)")

    parser.add_argument('--startframe',
                        default="5",
                        type=int,
                        help="specify startframe in Video. Default is 5 as the first frames are often broken in videos")

    args = parser.parse_args()
 
    if args.image:
       rawimage = cv2.imread(args.file,1)
    else:
       cap = cv2.VideoCapture(args.file)
       for counter in range(args.startframe):
          ret, rawimage = cap.read() 
       cap.release()

    if args.noundistort: 
       uimage = rawimage
    else:
       uimage = Undistort.UndistortImage(rawimage)

    #timeme = timeit.Timer('findWarpPoints(uimage,args.speedup,args.padding,args.debug)','from __main__ import findWarpPoints, uimage, args').timeit(number=100)
    #print ("time:", timeme)
    
    videoWarpPoints,Matrix,rotation = findWarpPoints(uimage,args.speedup,args.padding,args.debug)
    print ("bestwarppoints: [[%d,%d],[%d,%d],[%d,%d],[%d,%d]]" % (videoWarpPoints[0][0],videoWarpPoints[0][1],videoWarpPoints[1][0],videoWarpPoints[1][1],videoWarpPoints[2][0],videoWarpPoints[2][1],videoWarpPoints[3][0],videoWarpPoints[3][1]))
    print ("Board must be rotated by %d degrees." % ((rotation +1) * 90))
    
