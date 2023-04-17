import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.fft import fftfreq
from scipy.fft import fft
from PIL import Image
from scipy.optimize import curve_fit



files = np.sort(glob.glob('Sample_image/test_grid_dist*'))
print(files)


def get_peak_highest_prominence(array):
    peaks = find_peaks(array)[0]
    if peaks.size == 0:
        return []
    prom = peak_prominences(array, peaks)[0]
    max_prom = np.argmax(prom)
    return np.array(peaks[max_prom])

def search_peaks(array, show_FFT=False):
    '''
    Finds peaks in an 1darry.
    
    Parameters
    ----------
    
    array : 1darray
        Input array with peaks.
        
    p : number, optional
        The minimum prominence of the peaks. The prominence of a peak measures 
        how much a peak stands out from the surrounding baseline of the signal 
        and is defined as the vertical distance between the peak and its lowest 
        contour line.
        
    Returns
    -------
    peaks : 1darray
        Peaks of the array.
        
    peaks_index : 1darray
        Indices of the peaks.
    '''
    # All peaks should be above the mean
    h = np.mean(array[array!=0])

    
    # distance estimation via distance of peaks
    # one gets the distance by taken a fast fourier transform
    N = np.size(array) 
    t = np.linspace(0., N-1, N)
    f = fftfreq(len(t), np.diff(t)[0])
    x1_FFT = fft(array[array!=0])
    FFT = np.abs(x1_FFT[:N//2])

    
    peak = get_peak_highest_prominence(FFT)

    if np.size(peak) == 0:
        return [], []
    freq = f[peak]
    
    d = 1/freq * 0.8        # it's min distance, 
                            # take 80% of distance to account for fluctuation

    peak_information = find_peaks(array, height=h, 
                                    distance=d)

    peaks_index = peak_information[0]
    peaks = array[peaks_index]

    if show_FFT:
        plt.cla()
        plt.plot(f[:N//2], FFT)
        plt.vlines(freq, ymin=0, ymax=np.max(FFT), color='red')
        plt.show()
        
    return peaks, peaks_index

def search_maxima(array, show_FFT=False):
    '''
    Finds the local maxima of an 1darray.
    
    Parameters
    ----------
    array : 1darray
        Input array with local maxima.
        
    Returns
    -------
    maxima : 1darray, optional
        Maxima of the array.
        
    maxima_index : 1darray
        Indices of the Maxima.
    '''
    maxima, maxima_index = search_peaks(array, show_FFT)
    return maxima, maxima_index 

def get_lines_across_one_direction(img, axis=0, show_lines = False):
    """In a grid image, this gets the approximate prosition of the lines
    by integrating over one direction and looking for the peaks.

    Args:
        img (2D array): grid image
        axis (int, optional): Axis over the summation. Defaults to 0.
        show_lines (bool, optional): For debugging. Defaults to False.

    Returns:
        Array of int: Position of the peaks in img across the axis
    """
    sum = np.sum(img, axis=axis)
    peaks, peaks_index = search_maxima(-sum)
    if show_lines:
        plt.cla()
        plt.plot(-sum)
        plt.plot(peaks_index, peaks,'+')
        plt.show()
    return peaks_index


def only_number_of_peaks(array, peaks, peaks_idx, number_of_peaks=1):
    """Keeps only #number of peaks with highest prominence

    Args:
        array (1D array): Dataset containing the peaks
        peaks (1D array): Height of the peaks in array
        peaks_idx (1D array): index position of peak in array
        number_of_peaks (int, optional): How many peaks should be left. Defaults to 1.

    Returns:
        1D array: array containing of size #number with peak heights
        1D array: array containing of size #number with peak indexes.
    """
    while np.size(peaks) > number_of_peaks:
        prominences = peak_prominences(array, peaks_idx)[0]
        idx = np.argmin(prominences)
        peaks = np.delete(peaks, idx)
        peaks_idx = np.delete(peaks_idx, idx)
    return peaks, peaks_idx


def get_center_of_cross(img, show_center=False, show_sum=False):
    """Returns the position of a crossing point of two lines.
    The img should only contain a single cross

    Args:
        img (2D array): image that only contains 2 crossing lines.
        show_center (bool, optional): Debugging. Show the center points. Defaults to False.
        show_sum (bool, optional): Debugging. Show the summation over 2 axis. Defaults to False.

    Returns:
        list: [x coordinate, y coordinate] of crossing point
    """
    xsum = np.sum(img, axis=0)
    ysum = np.sum(img, axis=1)

    
    hx = np.mean(-xsum[xsum!=0])
    hy = np.mean(-ysum[ysum!=0])
    x_peaks_index = find_peaks(-xsum, height=hx)[0]
    y_peaks_index = find_peaks(-ysum, height=hy)[0]

    x_peak, x_peak_idx = only_number_of_peaks(-xsum, -xsum[x_peaks_index], x_peaks_index, number_of_peaks=1)
    y_peak, y_peak_idx = only_number_of_peaks(-ysum, -ysum[y_peaks_index], y_peaks_index, number_of_peaks=1)

    if show_sum:
        plt.cla()
        plt.plot(-xsum, color='tab:blue')
        plt.plot(-ysum, color='tab:orange')
        plt.plot(x_peak_idx, x_peak, '+')
        plt.plot(y_peak_idx, y_peak, '+')
        plt.show()
        
    if show_center:
        plt.cla()
        plt.imshow(img)
        plt.plot(x_peak_idx, y_peak_idx, '+', color='blue')
        plt.show()
    return [x_peak_idx, y_peak_idx]



def crop_image_with_padding(img, xc, yc, size):
    """ Crops an image to xc-size:xc+size and yc-size:yc+size. 
    But when any bound falls outside the image, the crop is filled with zeros.
    This will keep the size of the crop same for all crops.
    Padding is elegant here to maintain xc and yc the center.

    Parameters
    ----------
    img : 2d array
        Image that shall be cropped
    xc  : int
        x center of the crop
    yc  : int
        y center of the crop
    size: int
        The final image will have a size of 2*size,2*size

    Returns
    -------
    2d array
        cropped image which may contain zero
    """
    img_pad = np.pad(img, ((size,size),(size,size)), 'constant')
    xc = xc+size
    yc = yc+size
    x_low = xc-size
    x_high = xc+size
    y_low = yc-size
    y_high = yc+size

    return img_pad[y_low:y_high, x_low:x_high]


def cornersubPix(img, points):
    """Takes points, where the crosses of a grid image img are
    and returns a better, improve position of points.

    Args:
        img (2D array): grid image
        points (_type_): original guess for grid points

    Returns:
        2D array: improved guess for grid points
    """
    q,w = np.shape(points)
    re_cross = np.asarray(np.reshape(points, (q,1,w) ), dtype=np.float32) 
    gray = np.asarray(img, dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    re_corners = cv2.cornerSubPix(gray,re_cross, (1,1), (-1,-1), criteria)
    return re_corners.reshape((q,w))


def get_shape_of_grid_pattern(points, show_diff=False):
    """Determine the shape of the grid pattern. Additionally,
    this will print if they are any missing points

    Args:
        points (2d array): grid points
        show_diff (bool, optional): _description_. Defaults to False.

    Returns:
        int, int: shape of grid pattern
    """
    x = points[:,0,0]
    y = points[:,0,1]

    
    ydiff = np.abs(np.diff(y))
    h = np.max(ydiff)*0.5
    ypi = find_peaks(ydiff, height=h)[0]
    
    if show_diff:
        plt.cla()
        plt.plot(ydiff)
        plt.plot(ypi, ydiff[ypi], '+')
        plt.show()
    yshape = np.size(ypi)+1
    xshape = np.shape(points)[0] // yshape

    if np.mod(np.shape(points)[0], yshape):
        print('there are missing points') 
    return xshape, yshape



# I loop through all files to get the distortion correction
for i in files:
    print(i)
    filename = i.split('/')[-1]
    img = np.asarray(Image.open(i), dtype=float)
    if np.size(np.shape(img)) > 2:
        img = img[:,:,0]

    #img = align_grid_target(img)
    plt.cla()
    plt.imshow(img)
    plt.title('Raw image')
    plt.savefig('Results/' + filename + '_raw.png', dpi=300)
    #plt.show()
     
    # Determination of image points
    # loop through all expected grid points and crop the image
    # around these points with 2x square_size X 2x square_size. 
    # This gives a much better estimate of the grid points. 
    x_lines = get_lines_across_one_direction(img, axis=0)
    y_lines = get_lines_across_one_direction(img, axis=1)

    square_size = int(np.mean(np.diff(x_lines))*0.5)
    
    cross = []      #  stores all the grid points

    # This is needed to calculate the position of a found grid point
    # in the original image
    padx = square_size-x_lines[0]
    if padx < 0:
        padx=0
    pady = square_size-y_lines[0]
    if pady < 0:
        pady =0

    # loop through all lines in y and x direction to determine the grid points
    for j in y_lines:
        for k in x_lines:
            cropped_img = crop_image_with_padding(img, k, j, square_size)
            points = get_center_of_cross(cropped_img)

            # catches missing points
            if np.size(points[1])<1:
                continue
            if np.size(points[0])<1:
                continue

            a = points[0] + k - x_lines[0]-padx
            b = points[1] + j - y_lines[0]-pady
            cross.append([a[0],b[0]])
    

    cross = np.array(cross)     # Slicing with list does not work
    plt.cla()
    plt.plot(cross[:,0], cross[:,1], '+', color='tab:orange')
    plt.imshow(img, origin='lower')
    plt.title('Image with grid points')
    plt.savefig('Results/' + filename + '_gridpoints.png', dpi=300)
    #plt.show()

    # Reshape to make cv2 libraries work
    q,w = np.shape(cross)
        # Found out that cv2 needs np.float32
    re_cross = np.asarray(np.reshape(cross, (q,1,w) ), dtype=np.float32) 
    print(np.shape(re_cross))

    # Find better estimates of grid points
    # cv2 only works with uint8 type images
    imgcv2 = np.asarray(img/np.max(img)*(2**8-1), dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    re_corners = cv2.cornerSubPix(imgcv2,re_cross, (1,1), (-1,-1), criteria)

    # I remove the last two points to identify the order
    x = re_corners[:-2,0,0]
    y = re_corners[:-2,0,1]
    plt.cla()
    plt.imshow(imgcv2, origin='lower')
    plt.plot(x, y, '-', color='tab:orange')
    plt.plot(x, y, '+', color='red')
    plt.title('Refined grid points, last two points removed')
    plt.savefig('Results/' + filename + '_refindgridpoints.png', dpi=300)
    #plt.show()

    # create object points
    n, m = get_shape_of_grid_pattern(re_corners)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)

    print(np.shape(objp))

    plt.cla()
    plt.plot(re_corners[:-2,0,0], re_corners[:-2,0,1]-20, '-', color='tab:orange')
    plt.plot(re_corners[:-2,0,0], re_corners[:-2,0,1]-20, '+', color='red')
    plt.title('Object points, last two points removed')
    plt.savefig('Results/' + filename + '_objp.png', dpi=300)
    #plt.show()


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [re_corners], img.shape[::-1], None, None, flags=0)
    print('ret: ', ret, '\n\n', 
          'mtx: ', mtx, '\n\n', 
          'dist: ', dist, '\n\n', 
          'revec: ', rvecs, '\n\n', 
          'tvecs: ', tvecs) 

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    print('New camera matrix: ', newcameramtx)

    # undistort
    dst = cv2.undistort(imgcv2, mtx, dist, None, newcameramtx)
    print('Roi: ', roi)
    # crop the image
    x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]


    plt.cla()
    plt.imshow(dst, origin='lower')
    plt.title('Undistort - flag 0')
    plt.savefig('Results/' + filename + '_undistort_Flag0.png', dpi=300)
    #plt.show()

    # loop through some flags
    # Don't know what they do
    flags_working = []
    for i in range(20):
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [re_corners], img.shape[::-1], None, None, flags=2*i)
            flags_working.append(2*i)
        except:
            continue
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


        # undistort
        dst = cv2.undistort(imgcv2, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]

        # I found that if the mean of the image is below 95% of the mean of the original image
        # the image is generally very distorted and can be ignored
        if np.mean(dst)>np.mean(imgcv2)*0.95:
            plt.cla()
            plt.imshow(dst)
            plt.title('Undistort - Flag ' + str(i*2) + ' - mean: ' + str(np.mean(dst)))
            plt.savefig('Results/' + filename + '_undistort_Flag' + str(i*2) + '.png', dpi=300)
            #plt.show()