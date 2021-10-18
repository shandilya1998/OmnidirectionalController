import numpy as np
from utils.floquet_utils import *
pi = np.pi

class Floquet:
    def __init__(self, F, Jacobi, Tnum, T, omega, dt): # Method called automatically
        self.F = F
        self.Jacobi = Jacobi
        self.Tnum = Tnum
        self.dt = dt
        self.omega = omega
        self.T = T

    def Calc_X0_u0(self, X):
        X0_ = np.empty((X.shape[0],self.Tnum))
        u0_ = np.empty((X.shape[0],self.Tnum))
        for tt in range(self.Tnum):
            X0_[:,tt:tt+1] = X
            u0_[:,tt:tt+1] = self.F(X)/self.omega
            k1 = self.dt*self.F(X)
            k2 = self.dt*self.F(X+0.5*k1)
            k3 = self.dt*self.F(X+0.5*k2)
            k4 = self.dt*self.F(X+k3)
            X += (k1+2*k2+2*k3+k4)/6 # Runge Kutta
        return X0_, u0_

    def Calc_v0(self, X0_, rotations = 41):
        v0_ = np.empty((X0_.shape[0], X0_.shape[1])) # the same size as X
        v0_dif = np.empty((X0_.shape[0], X0_.shape[1])) # the same size as X
        v0 = np.ones((X0_.shape[0],1)) # initial point of v0(T)

        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array
                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6 # X_half_next = X(t-dt/2)
                X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T # X_next = X(t-dt)

                k1 = self.dt * np.dot(self.Jacobi(X).T, v0)
                k2 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k1/2) )
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T, (v0+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T, (v0+k3) )
                v0 = v0 + (k1+2*k2+2*k3+k4)/6 # RungeKutta method
                prob = np.dot(v0.T, self.F(X_next)/self.omega) # production <v0, u0>
            v0 = v0 / prob # normalization every cycle

        for tt in range(self.Tnum):#　storage
            X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] # list to array
            v0_[:, self.Tnum-tt-1:self.Tnum-tt] = v0 # storage backwards
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T

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
        SIZE = u0_.shape[0]
        u1 = np.ones((SIZE,1)) # initial point of u1(0)
        u1rec = np.empty((SIZE, self.Tnum)) # storage array
        norm1 = np.linalg.norm(u1)
        u1 = u1 / norm1 # normalization

        for rep in range(rotations): # run to convergence
            for tt in range(self.Tnum):
                prod1 = np.dot(v0_[:,tt:tt+1].T, u1) #　<u1, v0>
                u1 = u1 - prod1 * np.array([u0_[:,tt]]).T # remove u0 component
                X = X0_[:,tt:tt+1]

                h = self.dt/2
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T

                k1 = np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) * self.dt
                k2 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k1) * self.dt
                k3 = np.dot(self.Jacobi(X_half_next), u1 + 0.5*k2) * self.dt
                k4 = np.dot(self.Jacobi(X_next), u1 + k3) * self.dt
                u1 = u1 + (k1+2*k2+2*k3+k4)/6

            norm1 = np.linalg.norm(u1)
            u1 = u1 / norm1 # normalization
        lambda1 = np.log(norm1) / self.T

        for tt in range(self.Tnum): # storage
            u1rec[:,tt:tt+1] = u1
            prod1 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u1)
            u1 = u1 - prod1 * np.array([u0_[:,tt]]).T
            X = X0_[:,tt:tt+1]
            ### Runge ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T

            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt

            u1 = u1 + (k1+2*k2+2*k3+k4)/6

        return lambda1, u1rec

    def Calc_v1(self, lambda1, X0_, u0_, v0_, u1_, rotations = 2):
        SIZE = v0_.shape[0]
        v1 = np.ones((SIZE,1)) # initial point of v1(0)
        v1rec = np.empty((SIZE, self.Tnum)) # storage array

        for rep in range(rotations):
            for tt in range(self.Tnum):
                prod1 = np.dot( u0_[:,self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
                v1 = v1 - prod1 * np.array([v0_[:,self.Tnum-tt-1]]).T # remove v0 component
                X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array

                h = -1/2*self.dt
                k1 = h*self.F(X)
                k2 = h*self.F(X+0.5*k1)
                k3 = h*self.F(X+0.5*k2)
                k4 = h*self.F(X+k3)
                X_half_next = X + (k1+2*k2+2*k3+k4)/6
                X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T

                k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
                k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
                k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
                k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
                v1 = v1 + (k1+2*k2+2*k3+k4)/6

            v1 = v1 / np.dot(v1.T, u1_[:,self.Tnum-1:self.Tnum]) # <v1, u1> = 1

        for tt in range(self.Tnum):
            prod1 = np.dot( u0_[:,self.Tnum-tt-1:self.Tnum-tt].T, v1 ) # <v1, u0>
            v1 = v1 - prod1 * np.array([v0_[:,self.Tnum-tt-1]]).T # remove v0 component

            v1rec[:,self.Tnum-tt-1:self.Tnum-tt] = v1

            X = X0_[:,self.Tnum-tt-1:self.Tnum-tt] #list to array
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(self.Tnum-tt-2)%self.Tnum]]).T

            k1 = self.dt * (np.dot(self.Jacobi(X).T, v1) - lambda1 * v1)
            k2 = self.dt * (np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k1/2)))
            k3 = self.dt * np.dot(self.Jacobi(X_half_next).T - lambda1*np.identity(SIZE), (v1+k2/2) )
            k4 = self.dt * np.dot(self.Jacobi(X_next).T - lambda1*np.identity(SIZE), (v1+k3) )
            v1 = v1 + (k1+2*k2+2*k3+k4)/6

        return v1rec

    # monodromy matrix
    def monodromy(self, X0_):
        SIZE = X0_.shape[0]  #　state dimension
        M = np.identity(SIZE)  #　M is identity matrix
        for i in range(SIZE):
            y = M[:,i:i+1]  #　y is unit vector
            for tt in range(self.Tnum):  #　evol y for one period.
                # x(t+dt/2)
                X = X0_[:,tt:tt+1]
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
        SIZE = evec1.shape[0]  #　state dimension

        u1rec = np.empty((u1.shape[0], self.Tnum),dtype= complex) #保存用配列
        u2rec = np.empty((u2.shape[0], self.Tnum),dtype= complex) #保存用配列

        norm1 = np.linalg.norm(u1)
        norm2 = np.linalg.norm(u2)
        u1 = u1 / norm1 # normalization
        u2 = u2 / norm2

        for tt in range(self.Tnum):
            # save
            u1rec[:,tt:tt+1] = u1
            u2rec[:,tt:tt+1] = u2

            prod1 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u1) #　<u1, v0>
            prod2 = np.dot(np.conjugate(v0_[:,tt:tt+1].T), u2) # <u2, v0>
            u1 = u1 - prod1 * np.array([u0_[:,tt]]).T # remove u0 component
            u2 = u2 - prod2 * np.array([u0_[:,tt]]).T

            X = X0_[:,tt:tt+1]
            ### Runge Kutta ###
            h = self.dt/2
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = np.array([X0_[:,(tt+1)%self.Tnum]]).T

            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u1) - lambda1 * u1) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda1*np.identity(SIZE), u1 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda1*np.identity(SIZE), u1 + k3) * self.dt

            u1 += (k1+2*k2+2*k3+k4)/6

            k1 = ( np.dot(self.Jacobi(X0_[:,tt:tt+1]),u2) - lambda2 * u2) * self.dt
            k2 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k1) * self.dt
            k3 = np.dot(self.Jacobi(X_half_next) - lambda2*np.identity(SIZE), u2 + 0.5*k2) * self.dt
            k4 = np.dot(self.Jacobi(X_next) - lambda2*np.identity(SIZE), u2 + k3) * self.dt

            u2 += (k1+2*k2+2*k3+k4)/6
        return u1rec, u2rec

    def Calc_v1v2(self, lambda1, lambda2, evec1, evec2, X0_, u0_, v0_):
        v1 = np.copy(evec1) # v1(0) = v1(T)
        v2 = np.copy(evec2) # v2(0) = v2(T)
        SIZE = evec1.shape[0]  #　state dimension

        v1rec = np.empty((v1.shape[0], self.Tnum),dtype= complex) #保存用配列
        v2rec = np.empty((v2.shape[0], self.Tnum),dtype= complex) #保存用配列

        # v1,v2を計算

        for tt in range(self.Tnum):
            prod1 = np.dot( np.conjugate(np.array([u0_[:,(self.Tnum-tt)%self.Tnum]])), v1 ) # <u0, v1>
            prod2 = np.dot( np.conjugate(np.array([u0_[:,(self.Tnum-tt)%self.Tnum]])), v2 ) # <u0, v2>
            v1 = v1 - prod1 * np.array([v0_[:,(self.Tnum-tt)%self.Tnum]]).T # remove v0 component
            v2 = v2 - prod2 * np.array([v0_[:,(self.Tnum-tt)%self.Tnum]]).T

            X = np.array([X0_[:,(self.Tnum-tt)%self.Tnum]]).T
            h = -1/2*self.dt
            k1 = h*self.F(X)
            k2 = h*self.F(X+0.5*k1)
            k3 = h*self.F(X+0.5*k2)
            k4 = h*self.F(X+k3)
            X_half_next = X + (k1+2*k2+2*k3+k4)/6
            X_next = X0_[:,self.Tnum-tt-1:self.Tnum-tt]

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
            v1rec[:,self.Tnum-tt-1:self.Tnum-tt] = v1
            v2rec[:,self.Tnum-tt-1:self.Tnum-tt] = v2

        return v1rec, v2rec

    def product(self, y1_, y2_):
        IP = np.empty((self.Tnum)) # inner product
        for tt in range(self.Tnum):
            y1 = y1_[:,tt:tt+1]
            y2 = y2_[:,tt:tt+1]
            IP[tt] = np.abs(np.dot(np.conjugate(y1).T,y2))
            #IP[tt] = np.dot(np.conjugate(y1).T,y2).real
        return IP
