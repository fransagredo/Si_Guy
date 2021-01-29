#! /usr/bin/env python
import os, subprocess, shutil, re, numpy as np, scipy.io
import xml.dom.minidom as xml
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
#import h5py

# This toolkit is for QE calculations of gamma only! (automatic 1 1 1 does not hold!)
# This toolkit calculates the pw coefficients of wannier functions
har_to_eV = 27.21138

def construct_umatrix(umat_path):
    '''
    This function returns a numpy matrix constructed from a text file of Unk coefficients
    Specify the path of seedname_u.mat
    '''

    with open(umat_path, 'r') as umat:
        lines = umat.readlines()
    [num_kpts, num_wann, num_bands] = map(int, re.findall(r'\d+', lines[1]))

    u = np.empty([num_bands, num_wann], np.float64)
    row = 0
    col = 0

    for i in range(4,len(lines)):
        [z_real, z_img] = map(float, re.findall(r'-?\d+.?\d+',lines[i]))
        #z = complex(z_real, z_img)
        z = z_real
        u[row,col] = z
        row += 1
        if row == num_bands:
            col += 1
            row = 0

    return u

def construct_bloch_pw_hdf5(wfc_path, band_min, band_max):
    '''
    This function reads the pw coefficients between band_min and band_max (the ones included in the wannierization- usually all occupied bands)
    Minimum index for band_min is 1 and maximal index for band_max is the number of bands
    It returns a numpy array where each cell is a numpy array of a band pw coefficients
    '''

    with h5py.File(wfc_path, "r") as f:
        all_bands = f["evc"][()]
    bloch_pw_coeffs = all_bands[band_min-1:band_max]

    return bloch_pw_coeffs

def construct_bloch_pw_dat(wfc_path):
    '''
    Read the pw coeefcients of all bands from wfc_path
    '''

    with scipy.io.FortranFile(wfc_path, mode='r') as f:
        ik, xk, ispin, gamma_only, scalef = f.read_record('i4,3f8,i4,i4,f8')[0]
        ngw, igwx, npol, nbnd = f.read_record('4i4')
        b1, b2, b3 = f.read_record('3f8')
        itmp = f.read_record('{}i4'.format(3*igwx))

        wtmp = [None] * nbnd
        for ib in range(nbnd):
            wtmp[ib] = f.read_record('{}f8'.format(npol*igwx))
    
    all_bands = np.array(wtmp)
    b0 = all_bands[:,0,:]
    b1 = all_bands[:,1,:]
    all_bands = np.concatenate((b0,b1),1)
    return all_bands

def calc_wannier_pw_hdf5(umat_path, wfc_path, band_min, band_max):
    '''
    This function calculate the wannier pw coefficients
    It returns a numpy array where each cell is a numpy array of a band pw coefficients
    '''
    u = construct_umatrix(umat_path)
    bloch_pw = construct_bloch_pw_hdf5(wfc_path, band_min, band_max)

    if u.shape[0] != bloch_pw.shape[0]:
        print('Number of elements in a column in Unk is different from number of bands that were read from wfc')
        exit

    wannier_pw = np.matmul(u.transpose(),bloch_pw)
    return wannier_pw

def calc_wannier_energy(umat_path, xml_path, band_min, band_max):
    '''
    This function calculates the energies of all wannier functions according to e_wannier = sum over bands of |Ui|^2*epsilon_i (in eV)
    xml_path should be the path to the wannierized system
    '''
    u = construct_umatrix(umat_path)

    tree = et.parse(xml_path)
    root = tree.getroot()
    # get all kpoints
    kpoints = root.findall('./output/band_structure/ks_energies')
    # get the first and only kpoint
    kp = kpoints[0]
    # get a list of this kpoint eigenvalues
    eig = list(map(float, kp[2].text.split()))
    # take relevant eigenvalues
    eig = np.array(eig[band_min-1: band_max]) * har_to_eV

    e_wannier = np.matmul(np.power(u.transpose(),2), eig)
    
    return e_wannier

def wannier_max_energy(umat_path, xml_path, band_min, band_max):
    '''
    This function returns the index of the highest energy WF (between 1-num_wann)
    as well as the energy in eV
    '''
    
    e_wannier = calc_wannier_energy(umat_path, xml_path, band_min, band_max)
    max_e = max(e_wannier)
    max_index = np.argmax(e_wannier) + 1

    return max_e, max_index
