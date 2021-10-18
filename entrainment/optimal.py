import numpy as np
import matplotlib.pyplot as plt
from simulations.entrainment import phase, lagrange, \
    modhopf, integration, floquet
pi = np.pi
from constants import params
from tqdm import tqdm

def find_optimal_input(
    omega,
    P = 1e0,
    Delta = 0.0,
    degree = 15,
    Cpath = '../assets/out/plots/coef.npy'
):
    ND = integration.ND(params['tmax'], dt)

    # Van der Pol
    # di/dt = self.mu*i - i*i*i/3.0 - v + self.x0
    # dv/dt = i

    OMEGA = np.array([[omega]], dtype = np.float32)
    Tnum = (2 * np.pi / (OMEGA * dt))
    timescale = 1 # Transformation from t to t'=t/timescale
    ModHopf = modhopf.ModHopf_rescale(OMEGA, timescale)

    # inital point
    Xstart = np.ones((2,1)) 

    print("calculating X to convergence...", end="")

    # run to convergence
    Xstart = ND.evol_to_convergence(ModHopf.dif, Xstart)
    print("converged X = ", Xstart.T)

    # search period

    # The phase is 0 when the y falls below y_basis = 0.0
    Tnum, Xstart = ND.find_Tnum_Y(ModHopf.dif, Xstart, 0.0)
    T = dt * Tnum
    omega = 2*pi/T

    print("period T = ", T)
    print("frequency omega = ", omega)

    print("calculate limit cycle X0_ ...", end="")

    # calculate floquet vector
    Floquet = floquet.Floquet(ModHopf.dif, ModHopf.Jacobian, Tnum, T, omega, dt)

    # limit cycle X0_ and the first floquet vector u0
    X0_, u0_ = Floquet.Calc_X0_u0(Xstart)

    print("done")
    print("calculate v0_ ...", end="")

    # the first left floquet vector v0_ and v0_dif = d(v0)/dt
    # For convergence from any initial point of v0(0), we evol v0 for rotations = 10 times.
    v0_, v0_dif = Floquet.Calc_v0(X0_, 10)

    print("done")
    print("calculate u1_ ...", end="")

    # the second right floquet vector u1_ and floqet eigenvalue lambda1
    # For convergence from any initial point of u0(0), we evol u0 for rotations = 2 times.
    lambda1, u1_ = Floquet.Calc_u1(X0_, u0_, v0_, 2)

    print("done")
    print("calculate v1_ ...", end="")
    # the second left floquet vector v1_
    # For convergence from any initial point of v1(0), we evol v1 for rotations = 2 times.
    v1_ = Floquet.Calc_v1(lambda1, X0_, u0_, v0_, u1_, 2)

    print("done")
    print("")
    print("calculating floquet vector has finished")

    #%%
    ###### plot #######################################################
    ## verificate production of floquet vector
    fig = plt.figure(0, figsize=(8.0, 6.0))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    Time_ax = np.linspace(0,Tnum-1, Tnum)*dt

    fig.suptitle("Verification of floquet product")
    # <u0,v0>
    ax1.plot(Time_ax, Floquet.product(u0_, v0_))
    ax1.set_title('<u0,v0>')
    ax1.grid()
    # <u0,v1>
    ax2.plot(Time_ax, Floquet.product(u0_, v1_))
    ax2.set_title('<u0,v1>')
    ax2.grid()
    # <u0,v2>
    ax3.plot(Time_ax, Floquet.product(u1_, v0_))
    ax3.set_title('<u1,v0>')
    ax3.grid()
    # <u0,v0>
    ax4.plot(Time_ax, Floquet.product(u1_, v1_))
    ax4.set_title('<u1,v1>')
    ax4.grid()

    ## preparation for drawing and saving floquet vector #####################################

    figpath = "fig/ModHopf/Floquet/"
    os.makedirs(figpath, exist_ok=True) # make folder

    datapath = "data/ModHopf/Floquet/"
    os.makedirs(datapath, exist_ok=True) # make folder

    ## u0_ 
    filename = "u0_.pdf"
    filepath = figpath + filename

    plt.figure(1)
    plt.plot(Time_ax, u0_[0,:], label = r"$u_{0x}$", color = 'r')
    plt.plot(Time_ax, u0_[1,:], linestyle = '--', label = r"$u_{0y}$", color = 'g')
    plt.xlabel(r"$\theta$", size = 20)
    plt.ylabel(r"$u_0$", size = 20)

    plt.xticks([0, T/4, T/2, 3*T/4, T], [ "0", "π/2", "π", "3π/2", "2π"])
    #plt.xticks([0, T/4, T/2, 3*T/4, T], [ "0", "T/4", "T/2", "3T/4", "T"])
    plt.xlim(0,T)
    #plt.ylim(-1.3,1.3)
    plt.legend(fontsize=17)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0, transparent=True)

    ## u1_
    filename = "u1_.pdf"
    filepath = figpath + filename

    plt.figure(2)
    plt.plot(Time_ax, u1_[0,:], label = r"$u_{1x}$", color = 'b')
    plt.plot(Time_ax, u1_[1,:], linestyle = '--', label = r"$u_{1y}$", color = 'orange')
    plt.xlabel(r"$\theta$", size = 20)
    plt.ylabel(r"$u_1$", size = 20)

    plt.xticks([0, T/4, T/2, 3*T/4, T], [ "0", "π/2", "π", "3π/2", "2π"])
    plt.xlim(0,T)
    #plt.ylim(-1.3,1.3)
    plt.legend(fontsize=17)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0, transparent=True)

    ## v0_
    filename = "v0_.pdf"
    filepath = figpath + filename

    plt.figure(3)
    plt.plot(Time_ax, v0_[0,:], label = r"$v_{0x}$", color = 'r')
    plt.plot(Time_ax, v0_[1,:], linestyle = '--', label = r"$v_{0y}$", color = 'g')
    plt.xlabel(r"$\theta$", size= 20)
    plt.ylabel(r"$v_0$", size = 20)

    plt.xticks([0, T/4, T/2, 3*T/4, T], [ "0", "π/2", "π", "3π/2", "2π"])
    plt.xlim(0,T)
    #plt.ylim(-1.1,1.1)
    plt.legend(fontsize=17)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0, transparent=True)

    ## v1_
    filename = "v1_.pdf"
    filepath = figpath + filename

    plt.figure(4)
    plt.plot(Time_ax, v1_[0,:], label = r"$v_{1x}$", color = 'b')
    plt.plot(Time_ax, v1_[1,:], linestyle = '--', label = r"$v_{1y}$", color = 'orange')
    plt.xlabel(r"$\theta$", size = 20)
    plt.ylabel(r"$v_1$", size = 20)

    plt.xticks([0, T/4, T/2, 3*T/4, T], [ "0", "π/2", "π", "3π/2", "2π"])
    plt.xlim(0,T)
    #plt.ylim(-1.4,1.4)
    plt.legend(fontsize=17)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0, transparent=True)

    ## v0_dif
    filename = "v0_dif.pdf"
    filepath = figpath + filename

    plt.figure(5)
    plt.plot(Time_ax, v0_dif[0,:], label = r"$v_{0x}^{dif}$", color = 'r')
    plt.plot(Time_ax, v0_dif[1,:], linestyle = '--', label = r"$v_{0y}^{dif}$", color = 'g')
    plt.xlabel(r"$\theta$", size=20) 
    plt.ylabel(r"$v_0^{dif}$", size = 20) 
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0, transparent=True)

    ## save data
    X0_.dump(datapath + 'X0_.dat')
    u0_.dump(datapath + 'u0_.dat')
    u1_.dump(datapath + 'u1_.dat')
    v0_.dump(datapath + 'v0_.dat')
    v1_.dump(datapath + 'v1_.dat')
    v0_dif.dump(datapath + 'v0_dif.dat')


    Omega = omega - Delta
    Tenum = omega / Omega * Tnum # period with external force

    # calculate lagrange multipliers
    print("calculate lagrange multipliers ...")
    simple_2D = Lag.simple(Tnum, v0_, v0_dif, omega, P, Delta)
    mu, nu = simple_2D.Calc_mu_nu()
    print("nu = ", nu, " mu = ", mu)
    #%%
    Initial = 1/4 # initial phase (2π*Initial)
    X_phase = int(Tnum*Initial)
    x = np.copy(X0_[:,X_phase:X_phase+1])

    Tsimu = int(410/timescale) # simulation time
    Tsimu_num = int(Tsimu/dt)
    ModHopf_input = np.empty((2,int(Tenum))) # save input

    # save input
    for tt in range(int(Tenum)):
        #External Force Phase
        EFP = Omega * tt / omega
        EFP = EFP - int(EFP / Tnum) * Tnum
        IDP = EFP - int(EFP) # internally dividing point

        # Linear interpolation
        v0 = IDP * v0_[:,int(EFP):int(EFP)+1] + (1 - IDP) * np.array([v0_[:,(int(EFP)+1)%Tnum]]).T
        v0dif = IDP * v0_dif[:,int(EFP):int(EFP)+1] + (1 - IDP) * np.array([v0_dif[:,(int(EFP)+1)%Tnum]]).T

        # input
        q = 1/2/nu * (-1/omega * v0dif + mu * v0)

        # save
        ModHopf_input[:,tt:tt+1] = q

    #%%
    Gamma = simple_2D.Calc_Gamma(mu, nu)

    datapath = "data/ModHopf/simple/P{}Delta{}Initial{}/".format(P,Delta,Initial)
    os.makedirs(datapath, exist_ok=True) # make folder

    np.savetxt(datapath + 'mu_nu.txt', np.array([mu,nu]), delimiter = ',')
    ModHopf_input.dump(datapath + "input.dat")
    Gamma.dump(datapath + 'Gamma.dat')
