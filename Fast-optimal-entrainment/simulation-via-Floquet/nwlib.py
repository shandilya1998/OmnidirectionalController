import numpy as np 
import matplotlib.pyplot as plt
pi = np.pi

################ time integration ############################################
class ND:
    def __init__(self, tmax, dt): # Method called automatically
        self.tmax = tmax
        self.dt = dt
        self.t_ = np.arange(0, tmax, dt)
        self.tnum = len( self.t_ ) # Column length
    
    def evol(self, F, X):
        k1 = self.dt*F(X)
        k2 = self.dt*F(X+0.5*k1)
        k3 = self.dt*F(X+0.5*k2)
        k4 = self.dt*F(X+k3)
        X += (k1+2*k2+2*k3+k4)/6
        return X
        
    def evol_to_convergence(self, F, X):
        Tsimu_num = int(self.tnum)
        for tt in range(Tsimu_num):
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6 # Runge Kutta
        return X
    
    def find_Tnum_Y(self, F, X, y_basis = 0.0):
        m = 0
        n = 0
        y_temp = X[0,1]
        num = []
        #print('start')
        #fig, axes = plt.subplots(2, 1, figsize = (5, 10))
        #lst = []
        #print(self.dt)
        while m < 2:
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6
            #lst.append(X.copy())
            #axes[0].plot(np.stack(lst, 0)[:, 0, 0])
            #axes[1].plot(np.stack(lst, 0)[:, 1, 0])
            #plt.pause(0.05)
            if y_basis < y_temp and y_basis >= X[0,1]: # θ = 0 when the y component falls below y_basis.
                num.append(n)
                if m == 0:
                    Xstart = X
                m += 1
                #print('found')
            n += 1
            if n > 1e8:
                print("Limitcycle doesn't pass 'y_basis'")
                raise NotImplementedError
            y_temp = X[0,1]
        Tnum = num[1] - num[0] 
        #print('done')
        return Tnum, Xstart
    
    def find_Tnum_X(self, F, X, x_basis):
        m = 0
        n = 0
        x_temp = X[0,0]
        num = []
        while m < 2:
            k1 = self.dt*F(X)
            k2 = self.dt*F(X+0.5*k1)
            k3 = self.dt*F(X+0.5*k2)
            k4 = self.dt*F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6
            if x_basis > x_temp and x_basis <= X[0,0]: # θ = 0 when the x component exceeds x_basis.
                num.append(n)
                if m == 0:
                    Xstart = X # この時のXを初期値とする. 
                m += 1
            n += 1
            if n > 1e8:
                print("Limitcycle doesn't pass 'x_basis'")
                raise NotImplementedError
            x_temp = X[0,0]
        Tnum = num[1] - num[0] 
        return Tnum, Xstart

################ Calculate floquet vector (v0, u1, v0, ...)####################################
class Floquet:
    def __init__(self, F, Jacobi, Tnum, T, omega, dt): # Method called automatically
        self.F = F
        self.Jacobi = Jacobi
        self.Tnum = Tnum    
        self.dt = dt
        self.omega = omega
        self.T = T
    
    def Calc_X0_u0(self, X):
        X0_ = np.empty((self.Tnum, X.shape[-1]))
        u0_ = np.empty((self.Tnum, X.shape[-1]))
        for tt in range(self.Tnum):
            X0_[tt:tt+1] = X
            u0_[tt:tt+1] = self.F(X)/self.omega
            k1 = self.dt*self.F(X)
            k2 = self.dt*self.F(X+0.5*k1)
            k3 = self.dt*self.F(X+0.5*k2)
            k4 = self.dt*self.F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6 # Runge Kutta
        return X0_, u0_
    
    def Calc_v0(self, X0_, rotations = 41):
        v0_ = np.empty(X0_.shape)
        v0_dif = np.empty((X0_.shape)
        v0 = np.ones((1, X0_.shape[-1])) # initial point of v0(T)
        
        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                X = X0_[self.Tnum-tt - 1: self.Tnum - tt] #list to array
                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6 # X_half_next = X(t-dt/2)
                X_next = np.array([X0_[(self.Tnum-tt-2)%self.Tnum]]) # X_next = X(t-dt)
                
                k1 = self.dt * np.dot(self.Jacobi(X).T, v0)
                k2 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k1/2) )
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T, (v0+k3) )
                v0 = v0 + (k1+2*k2+2*k3+k4)/6 # RungeKutta method
                prob = np.dot(v0.T, self.F(X_next)/self.omega) # production <v0, u0>
            v0 = v0 / prob # normalization every cycle
                
        for tt in range(self.Tnum):#　storage
            X = X0_[self.Tnum-tt-1:self.Tnum-tt] # list to array
            v0_[self.Tnum-tt-1:self.Tnum-tt] = v0 # storage backwards
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[(self.Tnum-tt-2)%self.Tnum]]).T
            
            k1 = self.dt * np.dot(self.Jacobi(X).T, v0)
            k2 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k1/2) )
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T, (v0+k3) )
            v0_next = v0 + (k1+2*k2+2*k3+k4)/6
            v0dif = -(v0_next - v0)/self.dt
            v0_dif[:, self.Tnum-tt-1:self.Tnum-tt] = v0dif        
            v0 = v0_next
        return v0_, v0_dif
    
    def Calc_u1(self, X0_, u0_, v0_, rotations = 2):
        SIZE = u0_.shape[-1] 
        u1 = np.ones((1,SIZE)) # initial point of u1(0)
        u1rec = np.empty((self.Tnum, SIZE)) # storage array
        norm1 = np.linalg.norm(u1)
        u1 = u1 / norm1 # normalization

        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                prod1 = np.dot(v0_[tt:tt+1].T, u1) #　<u1, v0>
                u1 = u1 - prod1 * np.array([u0_[tt]]).T # remove u0 component
                X = X0_[tt:tt+1]

                h = self.dt/2
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[(tt+1)%self.Tnum]]).T
        
                k1 = np.dot(self.Jacobi(X0_[tt:tt+1]),u1) * self.dt
                k2 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k1) * self.dt
                k3 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k2) * self.dt
                k4 = np.dot(self.Jacobi(X_next), u1 + k3) * self.dt
                u1 = u1 + (k1+2*k2+2*k3+k4)/6
                
            norm1 = np.linalg.norm(u1)
            u1 = u1 / norm1 # normalization
        lambda1 = np.log(norm1) / self.T
        
        for tt in range(self.Tnum): # storage
            u1rec[tt:tt+1] = u1
            prod1 = np.dot(np.conjugate(v0_[tt:tt+1].T), u1)
            u1 = u1 - prod1 * np.array([u0_[tt]]).T 
            X = X0_[tt:tt+1]
            ### Runge ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[(tt+1)%self.Tnum]]).T
        
            k1 = ( np.dot(self.Jacobi(X0_[tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt
        
            u1 = u1 + (k1+2*k2+2*k3+k4)/6

        return lambda1, u1rec
    
    def Calc_v1(self, lambda1, X0_, u0_, v0_, u1_, rotations = 2):
        SIZE = v0_.shape[-1] 
        v1 = np.ones((1, SIZE)) # initial point of v1(0)
        v1rec = np.empty((self.Tnum, SIZE)) # storage array
        
        for rep in range(rotations):
            for tt in range(self.Tnum): 
                prod1 = np.dot( u0_[self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
                v1 = v1 - prod1 * np.array([v0_[self.Tnum-tt-1]]).T # remove v0 component
                X = X0_[self.Tnum-tt-1:self.Tnum-tt] #list to array
                
                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[(self.Tnum-tt-2)%self.Tnum]]).T
            
                k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
                k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
                v1 = v1 + (k1+2*k2+2*k3+k4)/6
            
            v1 = v1 / np.dot(v1.T, u1_[self.Tnum-1:self.Tnum]) # <v1, u1> = 1
        
        for tt in range(self.Tnum): 
            prod1 = np.dot( u0_[self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
            v1 = v1 - prod1 * np.array([v0_[self.Tnum-tt-1]]).T # remove v0 component
            
            v1rec[self.Tnum-tt-1:self.Tnum-tt] = v1
            
            X = X0_[self.Tnum-tt-1:self.Tnum-tt] #list to array
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[(self.Tnum-tt-2)%self.Tnum]]).T
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
            v1 = v1 + (k1+2*k2+2*k3+k4)/6
                
        return v1rec
    
    # monodromy matrix
    def monodromy(self, X0_):
        SIZE = X0_.shape[-1]  #　state dimension
        M = np.identity(SIZE)  #　M is identity matrix
        for i in range(SIZE):
            y = M[i:i+1]  #　y is unit vector
            for tt in range(self.Tnum):  #　evol y for one period. 
                # x(t+dt/2)
                X = X0_[tt:tt+1]
                h = 1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
            
                # Runge-Kutta(Jacobi)
                k1 = self.dt*np.dot(self.Jacobi(X), y)
                k2 = self.dt*np.dot(self.Jacobi(X_half_next), y + k1/2)
                k3 = self.dt*np.dot(self.Jacobi(X_half_next), y + k2/2)
                k4 = self.dt*np.dot(self.Jacobi(np.array([X0_[:,(tt+1)%self.Tnum]]).T), y + k3)
                y = y + (k1+2*k2+2*k3+k4)/6
        return M
    
    def Calc_u1u2(self, lambda1, lambda2, evec1, evec2, X0_, u0_, v0_):
        u1 = np.copy(evec1) # u1(0)
        u2 = np.copy(evec2) # u2(0)
        SIZE = evec1.shape[-1]  #　state dimension  
    
        u1rec = np.empty((self.Tnum, u1.shape[-1]),dtype= complex) #保存用配列
        u2rec = np.empty((self.Tnum, u2.shape[-1]),dtype= complex) #保存用配列
        
        norm1 = np.linalg.norm(u1)
        norm2 = np.linalg.norm(u2)
        u1 = u1 / norm1 # normalization
        u2 = u2 / norm2
        
        for tt in range(self.Tnum):
            # save
            u1rec[tt:tt+1] = u1
            u2rec[tt:tt+1] = u2
            
            prod1 = np.dot(np.conjugate(v0_[tt:tt+1].T), u1) #　<u1, v0>
            prod2 = np.dot(np.conjugate(v0_[tt:tt+1].T), u2) # <u2, v0>
            u1 = u1 - prod1 * np.array([u0_[tt]]).T # remove u0 component
            u2 = u2 - prod2 * np.array([u0_[tt]]).T 
            
            X = X0_[tt:tt+1]
            ### Runge Kutta ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[(tt+1)%self.Tnum]]).T
        
            k1 = (np.dot(self.Jacobi(X0_[tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt
            
            u1 += (k1+2*k2+2*k3+k4)/6
            
            k1 = ( np.dot(self.Jacobi(X0_[tt:tt+1]),u2) - lambda2 * u2) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda2*np.identity(SIZE), u2 + k3) * self.dt
        
            u2 += (k1+2*k2+2*k3+k4)/6
        return u1rec, u2rec
    
    def Calc_v1v2(self, lambda1, lambda2, evec1, evec2, X0_, u0_, v0_):
        v1 = np.copy(evec1) # v1(0) = v1(T)
        v2 = np.copy(evec2) # v2(0) = v2(T) 
        SIZE = evec1.shape[-1]  #　state dimension  
        
        v1rec = np.empty((self.Tnum, v1.shape[-1]), dtype= complex) #保存用配列
        v2rec = np.empty((self.Tnum, v2.shape[-1]), dtype= complex) #保存用配列
        
        # v1,v2を計算
        
        for tt in range(self.Tnum): 
            prod1 = np.dot( np.conjugate(np.array([u0_[(self.Tnum-tt)%self.Tnum]])), v1 ) # <u0, v1>
            prod2 = np.dot( np.conjugate(np.array([u0_[(self.Tnum-tt)%self.Tnum]])), v2 ) # <u0, v2>
            v1 = v1 - prod1 * np.array([v0_[(self.Tnum-tt)%self.Tnum]]).T # remove v0 component
            v2 = v2 - prod2 * np.array([v0_[(self.Tnum-tt)%self.Tnum]]).T
            
            X = np.array([X0_[(self.Tnum-tt)%self.Tnum]]).T
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = X0_[self.Tnum-tt-1:self.Tnum-tt]
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - np.conjugate(lambda1) * v1)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - np.conjugate(lambda1)*np.identity(SIZE), (v1+k3) )
            v1 +=  (k1+2*k2+2*k3+k4)/6
            
            k1 = self.dt * (np.dot(self.Jacobi(X).T, v2) - np.conjugate(lambda2) * v2)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - np.conjugate(lambda2)*np.identity(SIZE), (v2+k3) )
            v2 +=  (k1+2*k2+2*k3+k4)/6
            
            # Note that we save v1(t) from (t = T-dt) to (t = 0). 
            # v(T-dt) = v(-dt)
            v1rec[self.Tnum-tt-1:self.Tnum-tt] = v1
            v2rec[self.Tnum-tt-1:self.Tnum-tt] = v2
            
        return v1rec, v2rec
    
    def product(self, y1_, y2_):
        IP = np.empty((self.Tnum)) # inner product
        for tt in range(self.Tnum):
            y1 = y1_[tt:tt+1]
            y2 = y2_[tt:tt+1]
            IP[tt] = np.abs(np.dot(np.conjugate(y1).T,y2))
            #IP[tt] = np.dot(np.conjugate(y1).T,y2).real
        return IP

def _get_beta(x, C, degree):
    x = np.abs(x)
    X = np.concatenate([x ** p for p in range(degree, -1, -1 )], 0)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)

def _get_omega_choice(phi):
    return np.tanh(1e3 * (phi))

class ModHopf:
    def __init__(self, omega):
        self.omega = omega
        self.C = np.load('../../assets/out/plots/coef.npy')
        self.degree = 15
        beta = _get_beta(self.omega, self.C, self.degree)
        self.mean = np.abs(1 / (2 * self.beta * (1 - self.beta)))
        self.amplitude = (1 - 2 * self.beta) / (2 * self.beta * (1 - self.beta))

    def hopf_simple_step(self, omega, mu, z):
        x, y = np.split(z, 2, 0)
        r = np.sqrt(x * x + y * y)
        dx = ((mu - r * r) * x - omega * y)
        dy = ((mu - r * r) * y + omega * x)
        z = np.concatenate([dx, dy], 0)
        return z

    def dif(self, X):
        x, y = np.split(X, 2, 0)
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2
        F = self.hopf_simple_step(omega, 1.0, X)
        return F

    def dif_per(self, X, q):
        x = X[0, 0]
        y = X[0, 0]
        phi = np.arctan2(y, x)
        omega = self.omega * (
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q[0, 0]
        fy = (1.0 - r * r) * y + omega * x + q[1, 0]
        F = np.stack([fx, fy], 0)
        return F

    def dif_per1(self, X, q1):
        x = X[0, 0]
        y = X[0, 0]
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2 
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q1
        fy = (1.0 - r * r) * y + omega * x
        F = np.stack([fx, fy], 0)
        return F

    def dwdx(self, x, y): 
        return -self.omega * self.amplitude * 1e3 * y * ( 
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def dwdy(self, x, y): 
        return self.omega * self.amplitude * 1e3 * x * ( 
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def Jacobian(self, X): 
        x = X[0, 0]
        y = X[1, 0]
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2 
        f1x = 1 - 3 * x * x - y * y - y * self.dwdx(x, y)
        f1y = -2 * x * y - omega - y * self.dwdy(x, y)
        f2x = -2 * x * y + omega + x * self.dwdx(x, y)
        f2y = 1 - 3 * y * y - x * x + x * self.dwdy(x, y)
        #print(f1x.shape)
        #print(f1y.shape)
        #print(f2x.shape)
        #print(f2y.shape)
    
        J = np.concatenate([
            np.concatenate([f1x, f1y], 0).T,
            np.concatenate([f2x, f2y], 0).T
        ], 0)
        #print(J.shape)
        return J

class ModHopf_rescale:
    def __init__(self, omega, timescale):
        self.omega = omega
        #print(self.omega)
        self.C = np.load('../../assets/out/plots/coef.npy')
        self.degree = 15
        self.beta = _get_beta(self.omega, self.C, self.degree)
        #print(self.beta)
        self.mean = np.abs(1 / (2 * self.beta * (1 - self.beta)))
        self.amplitude = (1 - 2 * self.beta) / (2 * self.beta * (1 - self.beta))
        #print(self.mean)
        #print(self.amplitude)
        self.timescale = timescale

    def hopf_simple_step(self, omega, mu, z): 
        x, y = np.split(z, 2, -1)
        r = np.sqrt(x * x + y * y)
        dx = ((mu - r * r) * x - omega * y)
        dy = ((mu - r * r) * y + omega * x)
        z = np.concatenate([dx, dy], -1)
        return z

    def dif(self, X): 
        x, y = np.split(X, 2, -1)
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2
        F = self.hopf_simple_step(omega, 1.0, X)
        return F * self.timescale

    def dif_per(self, X, q): 
        x, y = np.split(X, 2, -1)
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2 
        F = self.hopf_simple_step(omega, 1.0, X)
        fx, fy = np.split(F, 2, -1)
        fx = fx + q[0]
        fy = fy + q[1]
        F = np.concatenate([fx, fy], -1)
        return F * self.timescale

    def dif_per1(self, X, q1):
        x, y = np.split(X, 2, -1)
        phi = np.arctan2(y, x)
        omega = self.omega * ( 
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2 
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q1
        fy = (1.0 - r * r) * y + omega * x 
        F = np.concatenate([fx, fy], -1)
        return F * self.timescale

    def dwdx(self, x, y):
        return -self.omega * self.amplitude * 1e3 * y * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def dwdy(self, x, y):
        return self.omega * self.amplitude * 1e3 * x * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def Jacobian(self, X):
        x, y = np.split(X, 2, -1)
        phi = np.arctan2(y, x)
        omega = self.omega * (
            self.mean + self.amplitude * _get_omega_choice(phi)
        ) / 2
        f1x = 1 - 3 * x * x - y * y - y * self.dwdx(x, y)
        f1y = -2 * x * y - omega - y * self.dwdy(x, y)
        f2x = -2 * x * y + omega + x * self.dwdx(x, y)
        f2y = 1 - 3 * y * y - x * x + x * self.dwdy(x, y)
        #print(f1x.shape)
        #print(f1y.shape)
        #print(f2x.shape)
        #print(f2y.shape)
        
        J = np.concatenate([
            np.concatenate([f1x, f1y], -1).T,
            np.concatenate([f2x, f2y], -1).T
        ], -1) * self.timescale
        #print(J.shape)
        return J


################van der pol#################################################
class VAN:
    # di/dt = self.mu*i - i*i*i/3.0 - v + self.x0
    # dv/dt = i
    def __init__(self, mu, x0 = 1.0, y0 = 0.7):
        self.mu = mu
        self.x0 = x0
        self.y0 = y0
    
    def dif(self, X):
        x, y = np.split(X, 2, -1)
        fx = self.mu*x - x*x*x/3.0 - y + self.x0
        fy = x + self.y0
        F = np.concatenate([fx, fy], -1)
        return F
    
    def dif_per(self, X, q):
        x, y = np.splot(X, 2, -1)
        fx = self.mu*x - x*x*x/3.0 - y + self.x0 + q[0,0]
        fy = x + self.y0 + q[1,0]
        F = np.array([fx, fy], -1)
        return F
    
    def dif_per1(self, X, q1):
        x, y = np.split(X, 2 ,-1)
        fx = self.mu*x - x*x*x/3.0 - y + self.x0 + q1
        fy = x + self.y0
        F = np.concatenate([fx , fy], -1)
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        f1x = self.mu - x*x
        f1y = -1.0
        f2x = 1.0
        f2y = 0.0
        J = np.array([[f1x, f1y],
                      [f2x, f2y]])
        return J 
    
class VAN_rescale:
    # timescaling  
    def __init__(self, mu, timescale, x0 = 1.0, y0 = 0.7):
        self.mu = mu
        self.x0 = x0
        self.y0 = y0
        self.timescale = timescale
    
    def dif(self, X):
        x, y = np.split(X, 2, -1)
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0)
        fy = self.timescale * (x + self.y0)
        F = np.concatenae([fx, fy], -1)
        return F
    
    def dif_per(self, X, q):
        x, y = np.split(X, 2, -1)
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0) + q[0,0]
        fy = self.timescale * (x + self.y0) + q[1,0]
        F = np.concatenate([fx, fy], -1)
        return F
    
    def dif_per1(self, X, q):
        x, y = np.split(X,  2, -1)
        fx = self.timescale * (self.mu*x - x*x*x/3.0 - y + self.x0) + q
        fy = self.timescale * (x + self.y0)
        F = np.concatenate([fx, fy], -1)
        return F
    
    def Jacobian(self, X):
        x = X[0,0]
        f1x = self.timescale*(self.mu - x*x)
        f1y = -self.timescale
        f2x = self.timescale
        f2y = 0.0
        J = np.array([[f1x, f1y],
                      [f2x, f2y]])
        return J

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