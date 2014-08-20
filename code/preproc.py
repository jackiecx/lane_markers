import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from skimage import filter
from sklearn.linear_model import LinearRegression as LR

def find_horizon(img):
    m,n,o = img.shape
    for i in xrange(795):
        bins = np.arange(0,255,20)
        X = np.digitize(img[795-i:800-i,:,0].flatten(), bins)

        if len(np.where(X >= 11)[0]) >= 1500:
            img[:895-i,:,:] = 0
            cutoff = 895-i
            break
    return img, cutoff

def canny_filter(img):
	edges = filter.canny(img[:,:,0],sigma=3)

	return np.array([edges.T, edges.T, edges.T]).T

def split_image(img):
	return img[:,:800,:], img[:,800:,:]

def linear_reg(img, cutoff):
    newimg = img[cutoff+1:,:,:]
    newimg = newimg[-350:-150,:,:]
    y = np.where(newimg == True)[0]
    x = np.where(newimg == True)[1]
    
    lr = LR()
    lr.fit(x[:,np.newaxis],y)
    
    coef = lr.coef_
    intercept = lr.intercept_
    print coef
    print intercept
    
    x_hat = np.arange(-1000,2600,1)[:,np.newaxis]
    
    y_hat = lr.predict(x_hat)
    x_intercept = np.argmin(y_hat**2)
    plt.figure()
    plt.plot(x,y,'b.')
    plt.plot(x_hat,y_hat,'r-')
    return x_hat,y_hat