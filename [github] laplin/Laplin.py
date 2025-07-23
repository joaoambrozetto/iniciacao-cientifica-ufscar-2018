""" Laplace inverse transform applied to NMR
**********************************************************
Joao Teles - DCNME/UFSCar - Nov 2016 - jocoteles@gmail.com
Last update: Feb 15 2017

Some conventions:
- Coordinate names receive the variables x(X) and y(Y)
- x -> time in the direct space
- X -> time in the inverse space
- y -> amplitude in the direct space
- Y -> amplitude in the inverse space
- inv-space -> inverse space
- dir-space -> direct space

Code convenctions:
- Class names: compound names with uppercase initials like "MultiGauss" class.
- Method and funtion names: compound names in lower-uppercase order like "genData" method.
- variable names: all lowercase for more than 3 letters. Any case, otherwhise.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pdb import set_trace

def gaussLN(t, tc, dt, amp):
    """ Gaussian function in ln scale
    - t: time value
    - tc: distribution center
    - dt: distribution standard deviation in Np units (dt=ln(t_sigma/tc))
    - amp: distribution amplitude """
    return amp*np.exp(-0.5*((np.log(t)-np.log(tc))/dt)**2)

def funcIR(x, a, b, c):
    """ Inversion recovery function with offset. """
    return a*(1.0-2.0*np.exp(-x/b)) + c

def norm(a):
    """ Normalize list a to 1. """
    return [y/max(a) for y in a]

def maxX(a):
    """ Returns the x element where y is maximum.
    - a: data object with a.x and a.yn lists."""
    return a.x[a.yn.index(max(a.yn))]

def rmsDiff(Y1, Y2):
        """ Function to calculate the rms value of Y2 - Y1
        -> Returns (type: float)"""
        N = len(Y1)
	diff = [Y2[i]-Y1[i] for i in range(N)]
        power = reduce(lambda a, b: a + b*b, diff, 0)/N        
        return np.sqrt(power)

class LogGaussFit(object):    
    def __init__(self, invDist):
        """Class to fit inv-space time distributions with
        log-gauss curves.
        - invDist: inv-space distribution data object instance of
        the classes: InvLaplace, MultiGauss, ..."""        
        self.X = invDist.X          #list of inv-space time values
        self.Y = invDist.Y          #list of inv-space amplitude values
        self.Xf = None              #list of inv-space fitted time values
        self.Yf = None              #list of inv-space fitted amplitude values
        self.popt = None            #list of optimum parameters
        self.pfit = None            #list of tuples of optimum parameters and errors        
        self.ng = None              # number of log-gauss components
        self.gaussLN = None         # log-gauss multi component function used to fit
    def gaussLN1(self, t, *p):
        """ Single gaussian function in ln scale
        - t: time value
        - p: parameters list, where:        
        - p[0] = tc: distribution center
        - p[1] = dt: distribution standard deviation in Np units (dt=ln(t_sigma/tc))
        - p[2] = amp: distribution amplitude """
        return p[2]*np.exp(-0.5*((np.log(t)-np.log(p[0]))/p[1])**2)
    def gaussLN2(self, t, *p):
        """ Double gaussian function in ln scale
        - t: time value
        - p: parameters list, where:
        - p[i*3] = tc: distribution center
        - p[i*3+1] = dt: distribution standard deviation in Np units (dt=ln(t_sigma/tc))
        - p[i*3+2] = amp: distribution amplitude
        - i: i-th gauss-log function (i = 0, 1)"""
        y = p[2]*np.exp(-0.5*((np.log(t)-np.log(p[0]))/p[1])**2)
        y += p[5]*np.exp(-0.5*((np.log(t)-np.log(p[3]))/p[4])**2)
        return y
    def gaussLN3(self, t, *p):
        """ Triple gaussian function in ln scale
        - t: time value
        - p: parameters list, where:        
        - p[i*3] = tc: distribution center
        - p[i*3+1] = dt: distribution standard deviation in Np units (dt=ln(t_sigma/tc))
        - p[i*3+2] = amp: distribution amplitude
        - i: i-th gauss-log function (i = 0, 1, 2)"""
        y = p[2]*np.exp(-0.5*((np.log(t)-np.log(p[0]))/p[1])**2)
        y += p[5]*np.exp(-0.5*((np.log(t)-np.log(p[3]))/p[4])**2)
        y += p[8]*np.exp(-0.5*((np.log(t)-np.log(p[6]))/p[7])**2)
        return y
    def setGaussLN(self, n):
        """ Sets the gaussLN function to be used for fitting. """
        if n == 1:
            self.gaussLN = self.gaussLN1
        elif n == 2:            
            self.gaussLN = self.gaussLN2
        elif n == 3:
            self.gaussLN = self.gaussLN3
    def dataFit(self, n, p0):
        """ Method to fit log-gauss functions to data.
        n: number of log-gauss functions (1 to 3)
        - p0: parameters guess list, where:        
        - p0[i*3] = tc: distribution center
        - p0[i*3+1] = dt: distribution standard deviation in Np units (dt=ln(t_sigma/tc))
        - p0[i*3+2] = amp: distribution amplitude
        - i: i-th gauss-log function (i = 0 ,..., n-1)"""
        X = np.array(self.X)
        Y = np.array(self.Y)
        p0 = np.array(p0)
        self.ng = n
        self.setGaussLN(n)                
        self.popt, pcov = curve_fit(self.gaussLN, X, Y, p0)
        perr = np.sqrt(np.diag(pcov))
        self.pfit = []
        for i in range(n):
            l = []
            for j in range(3):
                k = 3*i+j
                l.append((self.popt[k],perr[k]))
            self.pfit.append(l)
    def genY(self, N):
        """ Method to generate fitted curve with N points. """                
        Xi = self.X[0]
        Xf = self.X[len(self.X)-1]
        r = float(Xf/Xi)**(1.0/(N-1))
        X = [Xi*r**x for x in range(N)]
        self.Xf = X        
        Yf = [self.gaussLN(t, *self.popt) for t in X]
        self.Yf = Yf
        
class MultiGauss(object):    
    def __init__(self, invDist):
        """Class to simulate multi-gaussian distributions.                
        - distParam: list of distribution gaussian
            components [[tc0,dt0,amp0],[...],...].
        - tc: distribution center
        - dt: distribution standard deviation in Np units
        - amp: distribution amplitude """
        self.param = invDist        #list of gaussians parameters
        self.X = None               #list of inv-space time values
        self.Y = None               #list of inv-space amplitude values
    def genData(self, Xi, Xf, N):
        """Method to generate a logarithmic time data distribution
            in the inverse space.
        - Xi: initial time
        - Xf: finish time        
        - N: number of points
        -> Generates the self.X an self.Y (type: list)"""
        # Inv-space time values generation:
        r = float(Xf/Xi)**(1.0/(N-1))
        X = [Xi*r**x for x in range(N)]
        self.X = X
        # Inv-space amplitude values generation:
        Y = [0]*N        
        for g in self.param:    #add the multi gaussian contributions
            Y0 = [gaussLN(t,g[0],g[1],g[2]) for t in X]
            Y = [Y[i]+Y0[i] for i in range(N)]
        self.Y = Y/max(Y)   #normalization        
    def plotData(self):
        """Plots data in X log scale."""
        plt.plot(self.X, self.Y)
        plt.ylabel('amplitude')
        plt.xlabel('inverse time [s]')
        plt.xscale('log')

class NMRDecay(object):    
    def __init__(self, invDist, seq_type):
        """Class to simulate a simple NMR decay signal using a inverse
        space distribution.
        - invDist: inv-space distribution data object instance of
        the classes: multiGauss, ...
        - seq_type: 'cpmg', 'invrec' """
        self.X = invDist.X      # list of inv-space time values
        self.Y = invDist.Y      # list of inv-space amplitude values        
        self.seq_type = seq_type
        self.x = None           # list of dir-space time values
        self.y = None           # list of dir-space amplitude values with no noise
        self.yn = None          # list of dir-space amplitude with noise
        self.spower = None      # signal power with no noise
        self.npower = None      # noise power only
        self.snr = None         # signal to noise ratio
    def setX(self, x):
        """Method to set the time values in the direct space
        from list x.        
        -> Generates the self.x (type: list)"""
        self.x = x[:]
    def setYn(self, yn):
        """Method to set the noisy amplitude values in the direct space
        from list yn.        
        -> Generates the self.yn (type: list)"""
        self.yn = yn[:]
    def genX(self, xi, xf, N):
        """Method to generate the time values in the direct space.
        - xi: initial time
        - xf: finish time        
        - N: number of points
        -> Generates the self.x (type: list)"""
        # Direct space time values generation:
        dt = float(xf-xi)/(N-1)
        x = [xi+dt*i for i in range(N)]
        self.x = x
    def genLogX(self, xi, xf, N):
        """Method to generate the log time values in the direct space.
        - xi: initial time
        - xf: finish time        
        - N: number of points
        -> Generates the self.x (type: list)"""
        # Direct space time values generation:
        r = float(xf/xi)**(1.0/(N-1))
        x = [xi*r**t for t in range(N)]
        self.x = x
    def genY(self):
        """Method to generate a linear time zero-noise data
        distribution in the direct space.
        -> Generates the self.y (type: list)"""
        # Direct space time values generation:        
        Nj = len(self.X)
        Ni = len(self.x)
        y = [0]*Ni
        if self.seq_type == 'cpmg':
            for i in range(Ni):
                for j in range(Nj):
                    kij = np.exp(-self.x[i]/self.X[j])
                    y[i] += kij*self.Y[j] 
        elif self.seq_type == 'invrec':
            for i in range(Ni):
                for j in range(Nj):
                    kij = 1.0 - 2.0*np.exp(-self.x[i]/self.X[j])
                    y[i] += kij*self.Y[j]         
        self.y = y
    def powerCalc(self, y):
        """ Function to calculate the signal power average
        -> Returns (type: float)"""
        N = len(y)
        power = reduce(lambda a, b: a + b*b, y, 0)/N        
        return power
    def addGaussNoise(self, snr):
        """ Method to add a zero mean gaussian noise to dir-space amplitude
        (self.y) with snr (signal to noise ratio)."""
        self.snr = snr
        spower = self.powerCalc(self.y) #signal with no noise
        npower = spower/snr   #noise power
        N = len(self.y)
        noise = np.random.normal(0, npower**0.5, N) #gaussian noise with npower variance
        self.yn = [self.y[i]+noise[i] for i in range(N)]
        self.snrCalc()
    def addUniformNoise(self, snr):
        """ Method to add a zero mean uniform noise to dir-space amplitude
        (self.y) with snr (signal to noise ratio)."""
        self.snr = snr
        spower = self.powerCalc(self.y) #signal with no noise
        npower = spower/snr   #noise power        
        N = len(self.y)
        noise = np.random.uniform(0, (3*npower)**0.5, N) #uniform noise with npower variance
        self.yn = [self.y[i]+noise[i] for i in range(N)]
        self.snrCalc()
    def snrCalc(self):
        """Method to calculate the signal and noise power, and the snr."""        
        self.spower = self.powerCalc(self.y) #signal with no noise
        diff = [self.yn[i]-self.y[i] for i in range(len(self.y))]
        self.npower = self.powerCalc(diff)   #noise only        
        self.snr = self.spower/self.npower
    def plotData(self, shownoise = True, color = 'b-'):
        """Plots data in x linear scale."""        
        if (shownoise):
            plt.plot(self.x, self.yn, color)
            plt.ylabel('amplitude with noise')
        else:
            plt.plot(self.x, self.y, color)
            plt.ylabel('amplitude without noise')
        
        plt.xlabel('direct time [s]')

class InvLaplace(object):
    def __init__(self, dirDist, seq_type):
        """Class for the ILT linear system solver classes.
            - dirDist: dir-space distribution data object.
            - seq_type: 0 = CPMG, 1 = IR"""
        self.x = dirDist.x      # list of dir-space time values
        self.y = dirDist.yn     # list of dir-space amplitude values
        self.seq_type = seq_type        
        self.reg_method = None
        self.zero_method = None
        self.X = None           # list of inv-space time values
        self.Y = None           # list of inv-space amplitude values
        self.pen_matrix = None  # penalty matrix
        self.penalty = [1, 0, 0]  # list for penalty coefficients [n, g, c]: power 'n', gradient 'g', curvature 'c'        
        self.prec_TSVD = 1.0e-12      #treshold for TSVD truncation
        self.min_negative = -0.005   #limit for non-negativity iterations
        self.max_border = 1.0e-1      #limit for zero border iterations
        self.add_pen = 2.0           #value to be added to pen_matrix for non-negativity
    def genX(self, Xi, Xf, N):
        """Method to generate logarithmic time values (self.X)
            in the inverse space.
        - Xi: initial time
        - Xf: finish time       
        - N: number of points
        -> Creates self.X (type: list)"""
        # Inv-space time values generation:
        r = float(Xf/Xi)**(1.0/(N-1))
        X = [Xi*r**x for x in range(N)]        
        self.X = X        
    def genKernel(self):
        """ Function to calculate the kernel of the linear system.        
        -> return K (numPy matrix)"""
        Nj = len(self.X)
        Ni = len(self.x)
        K = np.zeros((Ni,Nj))
        if self.seq_type == 0:   #cpmg
            for i in range(Ni):
                for j in range(Nj):
                    K[i,j] = np.exp(-self.x[i]/self.X[j])
        elif self.seq_type == 1: #IR
            for i in range(Ni):
                for j in range(Nj):
                    K[i,j] = 1.0 - 2.0*np.exp(-self.x[i]/self.X[j])
        return K
    def genPenMatrix(self):
        """ Function to generate the penalty matrix accordingly with penalty method.
            -> returns penalty matrix (numPy matrix)"""
        Nj = len(self.X)
        M = np.zeros((Nj,Nj))
        #power penalty matrix construction:
        M += self.penalty[0]*np.eye(Nj)
        #gradient or first derivative penalty matrix:
        G = np.zeros((Nj,Nj))
        for i in range(Nj-1):
            G[i,i], G[i,i+1] = 1.0, -1.0
        M += self.penalty[1]*np.dot(G.transpose(),G)
        #curvature or second derivative penalty matrix:
        C = np.zeros((Nj,Nj))            
        for i in range(Nj-2):  #Based on Y. Gao et al. Journ. Mag. Res. 271 (2016)
            C[i+1,i], C[i+1,i+2] = 1.0, 1.0
            C[i+1,i+1] = -2.0
        M += self.penalty[2]*np.dot(C.transpose(),C)
        self.pen_matrix = M
        return M
    def invMatrix(self, A, reg_method):
        """ Function to perform the inversion of matrix A with the chosen
        regularization method reg_method.
        - A: numpy matrix to be inverted.        
        - reg_method: regularization method (0 or 1).
            0: inversion without TSVD (trunc. sing. val. decomp.)
            1: inversion with TSVD      
        -> Calculates inverted matrix (numPy matrix)"""        
        self.reg_method = reg_method
        if reg_method == 1: #TSVD option: Reference -> Marcel D'Euridice's thesis.
            U, s, V = np.linalg.svd(A, full_matrices = 0) #SVD
            Ut = U.transpose()
            Vt = V.transpose()
            si = 1/s
            j = 0
            while s[j] < self.prec_TSVD: #truncate procedure
                si[j] = 0.0
                j += 1                        
            Ai = np.dot(Vt, np.dot(np.diag(si), Ut))
        else: #No TSVD
            Ai = np.linalg.inv(A)        
        return Ai    
    def genY(self, penalty, reg_method, zero_method):
        """ Method to calculate the inv-space amplitude data (self.Y) solving
        the linear system inverse problem.
        - penalty: penalty coefficients list [n, g, c]: power 'n', gradient 'g', curvature 'c'
        - reg_method: regularization method (0 = no TSVD, 1 = TSVD)
        - zero_method: string for zeroing method:
            options: '0': no zero method
                     'nn0': nonnegativity
                     'nnx': nonnegativity and zero x border points,
                            where x is an integer greater than 0
                     'x': zero x border points.            
        -> Calculates self.Y (numPy matrix)"""
                
        self.penalty = penalty        
        self.reg_method = reg_method
        self.zero_method = zero_method
        
        K = self.genKernel()      #kernel of the linear system
        P = self.genPenMatrix()   #Penalty matrix        
        Kt = K.transpose()        
        
        A = np.dot(Kt,K) + P
        Ai = self.invMatrix(A, reg_method)        
        Y = np.dot(np.dot(Ai,Kt), self.y)
        #Y = Y/np.max(Y)     #normalization
        
        nn_flag = False
        x_flag = False          
        x = int(zero_method.replace('nn',''))
        if ('nn' in zero_method):
            nn_flag = True
        if (x > 0):
            x_flag = True

        while (nn_flag or x_flag):
            if ('nn' in zero_method):
                z = self.checkNegatives(Y)
                if (z != []):
                    nn_flag = True                                            
                    for i in range(len(z)):
                        #P[z[i],z[i]] += 2**P[z[i],z[i]]
                        P[z[i],z[i]] += self.add_pen
                else:
                    nn_flag = False
            if (x > 0):
                z = self.checkBorders(Y, x)
                if (z != []):
                    x_flag = True                    
                    for i in range(len(z)):
                        P[z[i],z[i]] += (P[z[i],z[i]])**self.add_pen
                else:
                    x_flag = False
            A = np.dot(Kt,K) + P
            Ai = self.invMatrix(A, reg_method)        
            Y = np.dot(np.dot(Ai,Kt), self.y)
            #Y = Y/np.max(Y)     #normalization        
        self.pen_matrix = P
        self.Y = [Y[i] for i in range(len(Y))]        
    def checkNegatives(self, Y):
        """Function to calculate Y array element positions with values
        smaller than self.min_negative
        - Y: numpy array
        -> Returns list with elment positions."""
        z = []
        min = self.min_negative*max(Y)
        for i in range(len(Y)):
            if Y[i] < min:
                z.append(i)
        return z
    def checkBorders(self, Y, x):
        """Function to calculate Y array element border positions with values
        greater than self.max_border
        - Y: numpy array
        - x: border size to be zeroed
        -> Returns list with elment positions."""
        z = []
        N = len(Y)-1
        max = self.max_border*max(Y)
        for i in range(x):
            if abs(Y[i]) > max:
                z.append(i)
            if abs(Y[N-i]) > max:
                z.append(N-i)        
        return z
    def plotData(self):        
        """Plots data in X log scale."""
        plt.plot(self.X, self.Y)
        plt.ylabel('amplitude')
        plt.xlabel('inverse time [s]')
        plt.xscale('log')

class ExpData(object):    
    def __init__(self):
        """Class to load dir-space experimental data."""        
        self.x = None           # list of dir-space time values        
        self.yn = None          # list of dir-space amplitude with noise        
        self.X = None           # list of inv-space time values        
        self.Y = None           # list of inv-space amplitude
    def loadFile(self, filename, col_x, col_y):
        """Method to load data from filename
        - col_x: column number corresponding to x coordinate.
        - col_y: column number corresponding to y coordinate."""
        f = open(filename, "r")
        line = f.readline()
        x = []
        yn = []
        while line:
            fields = line.split()
            x.append(float(fields[col_x-1]))
            yn.append(float(fields[col_y-1]))
            line = f.readline()
        f.close()
        self.x = x#np.array(x)
        self.yn = yn#np.array(yn)/max(yn)  #normalization
    def saveData(self, data, filename):
        """Method to save data to filename."""        
        self.X = data.X
        self.Y = data.Y
        f = open(filename, "w")
        for i in range(len(data.X)):
            s = str(data.X[i]) + "\t" + str(data.Y[i]) + "\n"
            f.write(s)
        f.close()
    def offsetRemove(self, x):
        """Method to remove offset from yn data.
        x: time to start considering the average offset calculation        
        -> Generates the self.yn offset corrected (type: list)"""
        i = 0        
        while self.x[i] < x:
            i += 1
        offset = 0.0
        N = len(self.x)
        for j in range(i, N):
            offset += self.yn[j]
        offset = offset/(N-i)
        for j in range(N):
            self.yn[j] -= offset        

class InvRecov(object):    
    def __init__(self, dirDist):
        """Class involved in inversion recovery data treatment."""        
        self.x = dirDist.x    # list of dir-space time values        
        self.yn = dirDist.yn   # list of dir-space amplitude with noise
        self.xf = None         # list of dir-space fitted time values                
        self.yf = None         # list of dir-space fitted amplitudes                
        self.popt = None        # list of optimum parameters
        self.perr = None        # list of errors in popt fitting
    def dataFit(self, T1):
        """ Method to fit IR data.
        T1: initial guess to longitudinal relaxation time."""
        x = np.array(self.x)
        yn = np.array(self.yn)
        self.popt, pcov = curve_fit(funcIR, x, yn, p0=(-self.yn[0], T1, 0.0))
        self.perr = np.sqrt(np.diag(pcov))
    def genY(self, N):
        """ Method to generate fitted curve with N points. """
        a, b, c = self.popt[0], self.popt[1], self.popt[2]
        ti = self.x[0]
        dt = (self.x[len(self.x)-1]-ti)/(N-1)
        self.xf = [ti+dt*i for i in range(N)]
        self.yf = [funcIR(t,a,b,c) for t in self.xf]

