import numpy as np
pi = np.pi

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
        y_temp = X[1,0]
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
            if y_basis < y_temp and y_basis >= X[1,0]: # θ = 0 when the y component falls below y_basis.
                num.append(n)
                if m == 0:
                    Xstart = X # この時のXを初期値とする.
                m += 1
                #print('found')
            n += 1
            if n > 1e8:
                print("Limitcycle doesn't pass 'y_basis'")
                raise NotImplementedError
            y_temp = X[1,0]
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
