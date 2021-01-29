import numpy as np
from scipy.special import erfc

def create_r2(n_max):
    r2 = np.zeros((2*n_max) **3 - 1)
    index = 0
    for i in np.arange(-n_max, n_max):
        for j in np.arange(-n_max, n_max):
            for k in np.arange(-n_max, n_max):
                if not( i== 0 and j ==0 and k==0):
                    r2[index]=i**2 + j**2 + k**2
                    index +=1
    return r2

def sr_madelung(gamma, leng, n_max = 5):
    """ this calculates the madelung constant due to the SR exact exchange term this is a negative number
    and should not be lower than -2.837"""
    # this calculates the next cubic shell of erfc terms to see if you are summing enough terms
    surface_term = n_max**2 * 6 * erfc(leng * gamma* n_max)
    if surface_term > 0.001:
        print('your grid is too small: ' +str(surface_term))
    r2 =create_r2(n_max)
    term1 = erfc(gamma *leng* r2**0.5)/ r2**0.5
    sum_term1 =np.sum(term1)
    background =  np.pi/ (gamma* leng)**2
    return sum_term1- background

def srsh_E_corr(q,alpha, gamma,onebyeps, leng):
    """ returns the energy correction for the given functional and cell size. q is the charge of the system. gamma is in ang^-1 and leng is in ang """
    sr_mad = -sr_madelung(gamma, leng)
    # converg leng to Bohr
    L =leng /0.529177 
    eV_per_hartree = 27.211386
    simp_cub_mad = 2.8373 #madelung constant for a simple cubic system
    # ecorr for a global hybrid functional with onebyeps exact exchange
    e_corr = simp_cub_mad *onebyeps* q**2/ (2 *L)# in Hartree
    sr_e_corr = sr_mad *(alpha-onebyeps)* q**2/ (2 *L)# in Hartree
    tot_e_corr = e_corr+ sr_e_corr
    Energy_corr =tot_e_corr * eV_per_hartree #in eV - this is max payne
    return Energy_corr
