import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

class simple:
    def __init__(self, Tnum, v0_, v0_dif, omega, P, Delta):
        self.Tnum = Tnum
        self.v0_ = v0_
        self.v0_dif = v0_dif
        self.omega = omega
        self.P = P
        self.Delta = Delta

    # Z = v0_  I1 = v1_
    # Z'(θ) = 1/ω dZ/dt
    def Calc_mu_nu(self):
        Z2 = 0 # <Z, Z>
        Zdif2 = 0 # <Z', Z'>
        #print(self.Tnum)
        #print(self.v0_.shape)
        for tt in range(self.Tnum):
            v0 = self.v0_[:,tt:tt+1]
            v0dif = self.v0_dif[:,tt:tt+1]
            #print(v0.shape)
            #print(Z2)
            #print(tt)
            Z2 = Z2 + np.dot(v0.T, v0) # <Z, Z>
            Zdif2 += np.dot(v0dif.T, v0dif) # <dZ/dt, dZ/dt>
        Z2 = Z2 / self.Tnum
        Zdif2 = Zdif2 / self.Tnum / self.omega / self.omega # <Z', Z'>
        if Zdif2 / (self.P - self.Delta**2 / Z2) < 0:
            print("Lagrange Error")
            print("cannot take the square root")
            sys.exit()
        nu = 1/2*np.sqrt( Zdif2 / (self.P - self.Delta**2 / Z2))
        mu = (- 2 * nu * self.Delta ) / Z2
        return mu.item(), nu.item()

    def Calc_mu_nu_arnold(self):
        Z2 = 0 # <Z, Z>
        Zdif2 = 0 # <Z', Z'>
        for tt in range(self.Tnum):
            v0 = self.v0_[:,tt:tt+1]
            v0dif = self.v0_dif[:,tt:tt+1]
            Z2 += np.dot(v0.T, v0) # <Z, Z>
            Zdif2 += np.dot(v0dif.T, v0dif) # <dZ/dt, dZ/dt>
        Z2 = Z2 / self.Tnum
        Zdif2 = Zdif2 / self.Tnum / self.omega / self.omega # <Z', Z'>
        if Zdif2 / (self.P - self.Delta**2 / Z2) < 0:
            return 0, 0 # not synchronization
        nu = 1/2*np.sqrt( Zdif2 / (self.P - self.Delta**2 / Z2))
        mu = (- 2 * nu * self.Delta ) / Z2
        return mu.item(), nu.item()

    def Calc_Gamma(self, mu, nu):
        # Γ(Φ) = ∫Z(Φ+ωt)・q(t)dt
        Division = 100 # Divide [-π, π] into 100 pieces.
        GammaPhi = np.empty((self.v0_.shape[0],Division+1))
        for tp in range(Division+1):
            Pnum = tp * int(self.Tnum/Division) - int(self.Tnum/2) # Φ(num) [-Tnum/2, Tnum/2]
            if(Pnum < 0):
                te = Pnum + self.Tnum # Φ = Pnum and Φ = Pnum + Tnum is equivalent for Z(Φ+ωt).
            else:
                te = Pnum
            # Calculate Γ(Φ)
            Gamma = 0
            for tt in range(self.Tnum):
                # Z0(Φ+ωt)
                v0phi = np.array([self.v0_[:,(tt+te)%self.Tnum]]).T
                # q(ωt)
                v0 = self.v0_[:,tt:tt+1]
                v0dif = self.v0_dif[:,tt:tt+1]
                q_ = 1/2/nu * (-1/self.omega * v0dif + mu * v0)
                # Γ(Φ) = ∫Z0(Φ+ωt)・q(ωt)dt
                Gamma += np.dot(q_.T, v0phi)
            # save
            GammaPhi[0,tp] = Pnum/self.Tnum*2*pi
            GammaPhi[1,tp] = self.Delta + Gamma/self.Tnum
        # plot
        plt.plot(GammaPhi[0,:], GammaPhi[1,:])
        plt.xlabel("Φ")
        plt.ylabel("Δ+Γ(Φ)")
        plt.xlim(-pi,pi)
        plt.grid()
        return GammaPhi
