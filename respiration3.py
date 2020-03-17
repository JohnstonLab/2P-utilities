import ScanImageTiffReader
import re
import fnmatch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import scipy.signal as signal
import fnmatch
from ScanImageTiffReader import ScanImageTiffReader as tiffread
from scipy import interpolate

def getTifListsFull(currdir):
    temp=[]           
    for file in os.listdir(currdir):
        if fnmatch.fnmatch(file, '*.tif'):
            temp.append(currdir+"/"+file)
    return temp


def getRespEventsAndAvg(I2C_array,p=(11,7,30,15), display=False,method="extremum", event="peak",rangeResp=0,parsing=[0,0]):
    """
    p = parameters
    event = "onset", "peak"
    method = "cwt", "extremum"
    """
    
    #display=False
    
    winSz=p[0] 
    polyOrder=p[1]
    widths=p[2]
    order=p[3]
    
    #############################
    
    x=I2C_array[:,0] #time in seconds
    y=I2C_array[:,1] #data
    
    y=signal.detrend(y)
    y = savitzky_golay(y, winSz, polyOrder)
    dy=np.diff(y)
    
    #############################
    if method=="extremum":
        if event == "peak":
            minima=argrelextrema(y, np.less,order=order)[0]
        if event == "onset":
            minima=argrelextrema(dy, np.less,order=order)[0]
            
    if method=="cwt":
        if event == "peak":
            minima=signal.find_peaks_cwt(-y,np.arange(1,widths))
        if event == "peak":
            minima=signal.find_peaks_cwt(-dy,np.arange(1,widths))

    #############################
    
    l=len(minima)
    interval=0
    for i in range(1,l-1): #boundaries cdts
        #print i
        interval+= minima[i+1]-minima[i]
    interval=interval/float(l-2)
    #freq=1/interval
    #print "interval: "+str(interval)
    #print "resp sampling rate: "+str(freq)
    r=int(interval/2)
    if rangeResp:
        r=rangeResp
    RespCycles=np.zeros((l-2,2*r))
    events=np.zeros(((l-2),3)) #start,event,end
#     print " " 
#     print "respRange: "+str(r)
#     print " "

    for i in range(1,l-1): #exclude the first one and last one: bound. cdts
        #print i
        RespCycles[i-1,:]=y[minima[i]-r:minima[i]+r]
        events[i-1,0]= I2C_array[minima[i]-r,0]
        events[i-1,1]= I2C_array[minima[i],0]
        events[i-1,2]= I2C_array[minima[i]+r,0]

    
    
    ############################# remove negative times
    #print events.shape
    #rint RespCycles.shape

    if events[0,0]<0 :
        zeroTime=np.where(events[:,0]<0.0)[0][-1]+1
        events=events[zeroTime:]
        RespCycles=RespCycles[zeroTime:]
    
    print(events.shape)
    #print RespCycles.shape
    ############################# parsing
    if (parsing != [0,0]).any():
        print("Parsing")
        startTime=np.where(events[:,0]>=parsing[0])[0][0]
        endTime=np.where(events[:,2]<=parsing[1])[0][-1]
        events = events[startTime:endTime+1]
        RespCycles=RespCycles[startTime:endTime+1]
        
        #print startTime
        #print endTime
        
    print(events.shape)
    #print RespCycles.shape
    ############################# get times
    
    temp=np.zeros(events.shape[0])
    for i in range(events.shape[0]): 
        temp[i] = events[i,2]-events[i,0]
        #print events[i,2]-events[i,0]

    intervalRangeResp = np.median(temp) #2B CHANGED TO MEAN; used here because outlier present due to blank frame/i2c
    
    
    avg=np.mean(RespCycles,axis=0)
    
    avgRespT=np.zeros(avg.shape[0])
    for i in range(avg.shape[0]):
        avgRespT[i] = intervalRangeResp*(i/ float(avg.shape[0]-1) )
        
    ############################# get frequency
    intervalCycleResp=0
    for i in range(1,events.shape[0]): 
        intervalCycleResp+= events[i,1]-events[i-1,1]#between 2 peaks
    intervalCycleResp = intervalCycleResp / (events.shape[0]-1)
    freq = 1 / intervalCycleResp
    
    #############################
    

    #print " "
    l=RespCycles.shape[0]
    if display:
        for i in range(l):
            #print i
            plt.plot(avgRespT,RespCycles[i])
        plt.plot(avgRespT, avg,linewidth=10,linestyle="--",color="black")
        plt.xlabel("seconds")
        
    return events, avgRespT, avg,freq
    

def getNumOfLines(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines

def getNumOfElem(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content] 
        numOfElem = len(content[0].split(" "))
    return numOfElem
	
def load2DArrayFromTxt(path):
    numOfLines = getNumOfLines(path)
    numOfElem = getNumOfElem(path)
    arr=np.zeros((numOfLines,numOfElem))

    with open(path) as f:
        content = f.readlines()
        content = [x.strip() for x in content] 

    for i in range(numOfLines):
        for j in range(numOfElem):
            arr[i,j]=float(content[i].split(" ")[j])
    return arr

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
	


def save2DArray2txt(arr,path,name): #arrayMetrics,path
    l=arr.shape[0]
    ll=arr.shape[1]
    fh = open(path+"/"+name+".txt","w") 
    for i in range(l):
        for j in range(ll):
            lines_of_text = [str(arr[i,j])+" "] 
            fh.writelines(lines_of_text) 
        lines_of_text = ["\n"] 
        fh.writelines(lines_of_text) 
    fh.close()

#def extractI2CfromSingleFile(path): ############ old
#    name=path.split("/")[-1][:-4]
#    savepath=path[:-len(path.split("/")[-1])]
#
#    I2C_list=[] #list of (timestamp, data, frame)
#
#    with ScanImageTiffReader.ScanImageTiffReader(path) as reader:
#        print(str(reader.shape())+"\n")
#        nOfFrames=reader.shape()[0]
#        for i in range(nOfFrames):
#            #print reader.description(i)
#            rawI2C = reader.description(i).split("\n")[14]
#            rawI2C_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", reader.description(i).split("\n")[14])[1:]
#            print(rawI2C_numbers)
#            numOfData = len(rawI2C_numbers)/2
#
#            if (numOfData > 0):
#                for j in range(numOfData):
#                    time=float(rawI2C_numbers[2*j])
#                    data=int(rawI2C_numbers[2*j+1])
#                    I2C_list.append((time,data,i+1))
#
#    I2C_array=np.asarray(I2C_list) #array of (timestamp, data, frame)
#    save2DArray2txt(I2C_array,savepath,name)
#    print("done "+name)
#    return I2C_array
    
def extractI2CfromSingleFile(path):
# path=fnames[0]

    name=path.split("/")[-1][:-4]
    savepath=path[:-len(path.split("/")[-1])]

    I2C_list=[] #list of (frame, timestamp, data-n)

    with ScanImageTiffReader.ScanImageTiffReader(path) as reader:
        print(str(reader.shape())+"\n")
        nOfFrames=reader.shape()[0]
        for i in range(nOfFrames):
            #print reader.description(i)
    #             rawI2C = reader.description(i).split("\n")[14]
            s = reader.description(i).split("\n")[14]
            rawI2C_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)[1:]
#             print(s)
#             print(">>>>>>>>")
#             print(rawI2C_numbers)
            ss=s.split("{{")[1].split("}")[0]
            lenSS=len(re.findall(r"[-+]?\d*\.\d+|\d+", ss))
            numOfData = int(len(rawI2C_numbers)/lenSS)
    #         numOfData = len(rawI2C_numbers)/2

            if (numOfData > 0):
                for j in range(numOfData):
                    pack=[]
                    time=float(rawI2C_numbers[lenSS*j])
                    pack.append(i+1)
                    pack.append(time)
                    for ii in range(lenSS-1):
                        data=int(rawI2C_numbers[lenSS*j+ii+1])
                        pack.append(data)
#                    print("pack, ",pack)
                    I2C_list.append(pack)

    I2C_array=np.asarray(I2C_list) #array of (frame, timestamp, data-n)
    save2DArray2txt(I2C_array,savepath,name)
    print("done "+name)
    return I2C_array
    
def extractI2CfromFilesDir(path):

# path=dirpath

    paths=[]
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, '*.tif'):
    #         print("file :",file)
            paths.append(path+"/"+file)
    #         print(paths[-1])

    name = path.split("/")[-1]
    savepath=paths[0][:-1-len(paths[0].split("/")[-1])]
    print(savepath)
    I2C_array = extractI2CfromSingleFile(paths[0])
    print(I2C_array.shape)
    print(paths[0]," done")

    for path in paths[1:]:
        newI2C_array = extractI2CfromSingleFile(path)
        newI2C_array[:,0] = newI2C_array[:,0] + I2C_array[-1,0]
        I2C_array = np.vstack((I2C_array,newI2C_array ) )
        print(I2C_array.shape)
        print(path," done")

    save2DArray2txt(I2C_array,savepath,name)
    print("done "+name)
    return I2C_array

def interpolateTime(x,t,newt,kind="zero",fill_value="extrapolate"):
#     kind="zero"#'previous' #kind of interpolation
#     #‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’,
#     nan = float('nan')
#     fill_value="extrapolate"#nan
    f_interp = interpolate.interp1d(t, x, kind=kind, fill_value=fill_value)
    xnew=f_interp(newt)
    return xnew

def getTimestampsScanImage(fname):
    ts=[]
    scanimage = tiffread(fname)
    for i in range(99999999):
        try:
            s=scanimage.description(i).split("frameTimestamps_sec ")[1]
            print(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0])
            ts.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0]))
        except:
            print(i, "last frame")
            break
    return np.asarray(ts)

def getTimestampsScanImageFromList(fnames):
    print(fnames[0])
    r=getTimestampsScanImage(fnames[0])
    print(">>>>", r.shape)
    for f in fnames[1:]:
        print(f)
        r=np.hstack((r,getTimestampsScanImage(f)))
        print(">>>>", r.shape)
    return r

def extractI2CfromFilesList(paths):

    path = ("/").join(paths[0].split("/")[:-1])
    name = path.split("/")[-1]
    savepath=paths[0][:-1-len(paths[0].split("/")[-1])]
    print(savepath)
    I2C_array = extractI2CfromSingleFile(paths[0])
    print(I2C_array.shape)
    print(paths[0]," done")

    for path in paths[1:]:
        newI2C_array = extractI2CfromSingleFile(path)
        newI2C_array[:,0] = newI2C_array[:,0] + I2C_array[-1,0]
        I2C_array = np.vstack((I2C_array,newI2C_array ) )
        print(I2C_array.shape)
        print(path," done")

    save2DArray2txt(I2C_array,savepath,name)
    print("done "+name)
    return I2C_array