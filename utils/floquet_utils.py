import numpy as np
pi = np.pi

################ arctan #####################################################
def arctan(x, y):
    if x >= 0 and y >= 0:
        theta = np.arctan(y/x)
    elif x < 0 and y >= 0:
        theta = np.arctan(y/x) + pi
    elif x < 0 and y < 0:
        theta = np.arctan(y/x) + pi
    elif x >= 0 and y < 0:
        theta = np.arctan(y/x) + 2*pi
    return theta

### Inner Product #########################################################
def inner_product(y1_, y2_, Tnum):
    IP = np.empty((Tnum))
    for tt in range(Tnum):
        y1 = y1_[:,tt:tt+1]
        y2 = y2_[:,tt:tt+1]
        IP[tt] = np.abs(np.dot(np.conjugate(y1).T,y2))
        #IP[tt] = np.dot(np.conjugate(y1).T,y2).real
    return IP
