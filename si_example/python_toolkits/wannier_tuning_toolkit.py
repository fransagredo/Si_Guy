#! /usr/bin/env python
import os, subprocess, shutil, re, numpy as np, math, scipy.io
import matplotlib.pyplot as plt
#import h5py
import wannier2pw as w2p, qe_toolkit as qe
import srsh_correction

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

ry_to_eV = 13.6057
madelung = 2.8373
hartree_to_ry = 2
ang_to_bohr = 1.8897

def is_normalized_hdf5(wfc_path, iband):
    '''
    WORKS ONLY FOR GAMMA ONLY CALCULATIONS!
    This function checks if the band iband is normalized
    It returns the value of <iband|iband>
    Minimum index is 1 and maximal index is the number of bands
    '''
    with h5py.File(wfc_path, "r") as f:
        all_bands = f["evc"][()]
    band_coeffs = all_bands[iband - 1]
    
    dot_product = np.dot(band_coeffs, band_coeffs)
   
    return dot_product

def normalize(v, normalize_to):
    '''
    This function normalizes the numpy vector v such that <v|v> = normalize_to
    '''
    norm = np.linalg.norm(v)
    norm_factor = math.sqrt(normalize_to)/norm
    v_normalized = norm_factor * v

    return v_normalized

def insert_wannier2wfc_hdf5(umat_path, wfc_orig_path, wfc_inserted_path ,new_wfc_path, band_min, band_max, ibloch, iwannier):
    '''
    This function replaces a single bloch band (ibloch) with a single wannier function (iwannier)
    Minimum index is 1 and maximal index is the number of bands
    wfc_orig_path is the wfc for which the wannierization was done
    wfc_inserted_path is the wfc into which we will insert the WF
    They can be the same, or different.
    A copy of the wfc into which we insert the WF is created anyway, and written in new_wfc_path
    band_min and band_max are the occupied bands from the wannierization (1-->HOMO)
    The function also normalizes the WF such that it is normalized to the original value that the bloch band was normalized to 
    '''
    # calc wannier coeffs
    wannier_pw = w2p.calc_wannier_pw_hdf5(umat_path, wfc_orig_path, band_min, band_max)
    wannier_coeffs = wannier_pw[iwannier-1]

    # normalize the wannier vector
    normalize_to = is_normalized_hdf5(wfc_inserted_path, ibloch)
    wannier_coeffs = normalize(wannier_coeffs, normalize_to)
        
    # read bloch coeffs of all bands from original wfc
    with h5py.File(wfc_inserted_path, "r") as f:
        all_bands = f["evc"][()]
        
        # replace relevant band bloch coeffs with wannier coeffs
        all_bands[ibloch-1] = wannier_coeffs

        # make a copy of the original wfc
        shutil.copyfile(wfc_inserted_path, new_wfc_path)

        # write the new data into the new wfc 
        with h5py.File(new_wfc_path, "r+") as new_wfc:
            del new_wfc["evc"]
            new_wfc.create_dataset("evc",data=all_bands)
            # add attributes and values (optional)
            for key, value in f["evc"].attrs.items():
                new_wfc["evc"].attrs[key] = value

def insert_wannier2wfc_dat(umat_path, wfc_orig_path, wfc_inserted_path ,new_wfc_path, band_min, band_max, ibloch, iwannier):
    '''
    Same but for dat files
    this function works only in python 2. disable import h5py
    '''

    # calc wannier coeffs
    u = w2p.construct_umatrix(umat_path)
    all_bands = w2p.construct_bloch_pw_dat(wfc_orig_path)
    wannierized_bands = all_bands[band_min-1:band_max]
    wannier_pw = np.matmul(u.transpose(),wannierized_bands)
    wannier_coeffs = wannier_pw[iwannier-1]

    # normalize the wannier vector
    replaced_band_coeffs = all_bands[ibloch-1]
    normalize_to = np.dot(replaced_band_coeffs, replaced_band_coeffs)
    wannier_coeffs = normalize(wannier_coeffs, normalize_to)

    # change the shape of wannier array
    cut = int(wannier_coeffs.shape[0]/2)
    w0 = wannier_coeffs[0:cut]
    w1 = wannier_coeffs[cut:]
    wannier_coeffs = np.empty([1,2,len(w0)])
    wannier_coeffs[0,0,:] = w0
    wannier_coeffs[0,1,:] = w1
    wannier_coeffs_list = np.ndarray.tolist(wannier_coeffs)

    # write the new data into the new wfc 

    with scipy.io.FortranFile(wfc_inserted_path, mode='r') as f, scipy.io.FortranFile(new_wfc_path, mode='w') as final:
        ik, xk, ispin, gamma_only, scalef = f.read_record('i4,3f8,i4,i4,f8')[0]
        ngw, igwx, npol, nbnd = f.read_record('4i4')
        b1, b2, b3 = f.read_record('3f8')
        itmp = f.read_record('{}i4'.format(3*igwx))
        final.write_record(ik, xk, ispin, gamma_only, scalef)
        final.write_record(ngw, igwx, npol, nbnd)
        final.write_record(b1, b2, b3)
        final.write_record(itmp)

        for ib in range(nbnd):
            if ib == (ibloch-1):
                final.write_record(wannier_coeffs_list)
            else:
                final.write_record(f.read_record('{}f8'.format(npol*igwx)))


def insert_wannier_and_copy_outdir(wannierization_dir, seedname, ihomo, nbnd, iwannier, outdir, new_outdir):
    '''
    This function creates a copy of QE 'calculations' outdir in new_outdir. Specify the path of the mother-directory of 'calculations' dir for both.
    It replaces the HUMO in wfcdw1 with the iwannier WF.   
    If 'calculations' already exists in new_outdir, it only replaces wfcdw1 in there.
    ihomo is the index of the homo, nbnd is the number of bands
    '''
    if os.path.exists(new_outdir+'calculations/')==False:
        shutil.copytree(outdir+'calculations/', new_outdir+'calculations/')
        
    umat_path = wannierization_dir+seedname+'_u.mat'
    wfc_orig_path = wannierization_dir+'calculations/'+seedname+'.save/wfc1.hdf5'
    wfc_inserted_path = new_outdir+'calculations/'+seedname+'.save/wfcdw1.hdf5'
    new_wfc_path = new_outdir+'calculations/'+seedname+'.save/wfcdw1_new.hdf5'
    band_min = 1
    band_max = ihomo
    ibloch = nbnd
    insert_wannier2wfc(umat_path, wfc_orig_path, wfc_inserted_path ,new_wfc_path, band_min, band_max, ibloch, iwannier)

    os.remove(wfc_inserted_path)
    os.rename(new_wfc_path, wfc_inserted_path)

def get_toten_and_epsilon_wannier(out_file_path):
    '''
    This function returns
    - total energy in ry
    - eig_wfcU of spin down channel in eV
    - n_wfcU (occupied) of spin down channel
    '''

    with open(out_file_path, 'r', errors='replace') as out:
        lines = out.readlines()

    # finding total energy
    for line in reversed(lines):
        match_toten = re.search(r'%', line)
        if match_toten:
            match = re.search(r'-?\d+.?\d+', line)
            toten_string = match.group()
            break

    # finding wannier eigenvalue
    for line in reversed(lines):
        match_eig = re.search(r'eig_wfcU', line)
        if match_eig:
            match = re.search(r'-?\d+.?\d+', line)
            eig_string = match.group()
            break

    # finding wannier projection over occupied states
    for line in reversed(lines):
        match_wann_proj_occ = re.search(r'n_wfcU \(occupied\)', line)
        if match_wann_proj_occ:
            match = re.search(r'-?\d+.?\d+', line)
            wann_proj_occ_string = match.group()
            break

    # finding wannier projection over all states
    for line in reversed(lines):
        match_wann_proj_all = re.search(r'n_wfcU \(all\)', line)
        if match_wann_proj_all:
            match = re.search(r'-?\d+.?\d+', line)
            wann_proj_all_string = match.group()
            break

    toten = float(toten_string)
    eig_wannier = float(eig_string)
    wann_proj_occ = float(wann_proj_occ_string)
    wann_proj_all = float(wann_proj_all_string)
    
    return toten, eig_wannier, wann_proj_occ, wann_proj_all

def get_input_srsh(in_file_path):
    '''
    This function returns:
    - gamma in bohr^-1
    - alpha
    - onebyeps
    - L in ang
    - lambda in ry
    '''

    with open(in_file_path, 'r', errors='replace') as inp:
        lines = inp.readlines()
    
    # finding gamma
    for line in lines:
        match_st = re.search(r'screening_parameter', line)
        if match_st:
            match = re.search(r'-?\d+.?\d+', line)
            gamma_string = match.group()
            break

    # finding alpha
    for line in lines:
        match_st = re.search(r'exx_fraction', line)
        if match_st:
            match = re.search(r'-?\d+.?\d+', line)
            alpha_string = match.group()
            break

    # finding beta
    for line in lines:
        match_st = re.search(r'rsh_beta', line)
        if match_st:
            match = re.search(r'-?\d+.?\d+', line)
            beta_string = match.group()
            break
            
    # finding a
    for line in lines:
        match_st_a = re.search(r' a = ', line)
        if match_st_a:
            match = re.search(r'-?\d+.?\d+', line)
            a_string = match.group()
            break

    # finding cell size
    for i in range(0,len(lines)):
        match_st = re.search(r'CELL_PARAMETERS alat', lines[i])
        if match_st:
            match = re.search(r'\d', lines[i+1])
            size_string = match.group()
            break

    # finding lambda
    for line in lines:
        match_st = re.search(r'lambda_wann_constr', line)
        if match_st:
            match = re.search(r'-?\d+.?\d+', line)
            lambda_string = match.group()
            break
    
    gamma_bohr = float(gamma_string)
    alpha = float(alpha_string)
    onebyeps = alpha + float(beta_string)
    if match_st_a:
        L_ang = float(a_string) * int(size_string)
    else:
        L_ang = 5 # This is a dummy value just so things will run
    if match_st:
        lambda_ry = float(lambda_string)
    else:
        lambda_ry = None

    return gamma_bohr, alpha, onebyeps, L_ang, lambda_ry
    
def calc_koopmans(gamma_dir_path, seedname, corr = 1, mp_eps = 0):
    '''
    This function calculates the wannier koopmans compliance: I = E(N-1)+ICC-E(N)+epsilon in eV
    for a single gamma directory. It assumes it has two subdirs: one has 'N_' and the other 'N-1_'
    use correction flag, corr, to determine which correction you would like to use:
    0 - no correction
    1 - makov-payne correction
    2 - SRSH correction
    anything else - will be regarded as the correction you want to add in eV

    mp_eps is the dielectric constant that will be used in the mp correction. if it is zero, the eps used is the one defined in the input files
    '''
    
    found_N_dir = False
    found_Nm1_dir = False

    # search for the N and N-1 directories
    for root, dirs, files in os.walk(gamma_dir_path):
        for dir0 in dirs:
            dir_name = os.path.join(dir0)
            match = re.search(r"(?:N_|N-1_)", dir_name)
            if match:
                if match[0] == 'N_':
                    N_dir = dir_name
                    found_N_dir = True
                elif match[0] == 'N-1_':
                    Nm1_dir = dir_name
                    found_Nm1_dir = True
    
    if not found_N_dir or not found_Nm1_dir:
        return None

    # pulling relevant values
    [E_N, eig_wann_N, wann_proj_occ_N, wann_proj_all_N] = get_toten_and_epsilon_wannier(gamma_dir_path+'/'+N_dir+'/'+seedname+'.out')
    [E_Nm1, eig_wann_Nm1, wann_proj_occ_Nm1, wann_proj_all_Nm1] = get_toten_and_epsilon_wannier(gamma_dir_path+'/'+Nm1_dir+'/'+seedname+'.out')
    [gamma_bohr_N, alpha_N, onebyeps_N, L_ang_N, lambda_ry_N] = get_input_srsh(gamma_dir_path+'/'+N_dir+'/'+seedname+'.in') 
    [gamma_bohr_Nm1, alpha_Nm1, onebyeps_Nm1, L_ang_Nm1, lambda_ry_Nm1] = get_input_srsh(gamma_dir_path+'/'+Nm1_dir+'/'+seedname+'.in') 

    # checking the parameters
    if gamma_bohr_N != gamma_bohr_Nm1:
        print('different gammas!',gamma_bohr_N,'vs',gamma_bohr_Nm1)
    if alpha_N != alpha_Nm1:
        print('different alphas!',alpha_N,'vs',alpha_Nm1)
    if onebyeps_N != onebyeps_Nm1:
        print('different onebyeps!',onebyeps_N,'vs',onebyeps_Nm1)
    if L_ang_N != L_ang_Nm1:
        print('different L!',L_ang_N,'vs',L_ang_Nm1)
    if wann_proj_all_N < 0.9998:
        print('NOTE: The N system sum of Wannier projections over all states is', wann_proj_all_N)
    if wann_proj_occ_Nm1 > 0.001:
        print('NOTE: The N-1 system sum of Wannier projections over occupied states is', wann_proj_occ_N)
    if wann_proj_all_Nm1 > 0.001:
        print('NOTE: The N-1 system sum of Wannier projections over all states is', wann_proj_occ_N)

    # image charge correction in ry
    srsh_corr = srsh_correction.srsh_E_corr(1, alpha_N, gamma_bohr_N * ang_to_bohr, onebyeps_N, L_ang_N)/ry_to_eV
    if mp_eps == 0:
        mp = (madelung*onebyeps_N/(2*ang_to_bohr*L_ang_N)) * hartree_to_ry
    else:
        mp = (madelung/(2*ang_to_bohr*L_ang_N*mp_eps)) * hartree_to_ry
    
    if corr == 0:
        ICC = 0
    elif corr == 1:
        ICC = mp
    elif corr == 2:
        ICC = srsh_corr
    else:
        ICC = corr/ry_to_eV

    # insert the wannier occupation to E(N-1)
    E_Nm1 = E_Nm1 + (lambda_ry_Nm1 * wann_proj_occ_Nm1)
    
    # koopmans compliance
    I = (E_Nm1 +ICC - E_N)*ry_to_eV + eig_wann_N
    
    return I

def get_koopmans(seedname, corr = 1, mp_eps = 0, plot = False):
    '''
    Print koopmans compliance from all directories in current directory
    '''
    
    all_dirs = []
    curr_dir = os.getcwd()
    for subdir, dirs, files in os.walk(curr_dir):
        for d in dirs:
            dir_path = curr_dir+'/'+d
            I = calc_koopmans(dir_path, seedname, corr, mp_eps)
            if I is not None:
               this_dir = {}
               this_dir['dir_name'] = d
               this_dir['val'] = qe.num_in_dir_name(d)
               this_dir['KC'] = I
               all_dirs.append(this_dir)

    #remove duplicates from dictionary
    all_dirs = [i for n, i in enumerate(all_dirs) if i not in all_dirs[n + 1:]]
        
    #sort the list of dictionaries by the 'val' key
    all_dirs_sorted = sorted(all_dirs, key = lambda i: i['val'])

    vals = []
    KCs = []
    # print
    for j in all_dirs_sorted:
        print(j['val'])
        print(j['KC'])
        print()
        vals.append(j['val'])
        KCs.append(j['KC'])

    # plot
    if plot:
        plt.plot(vals, KCs, color='k')
        plt.scatter(vals, KCs, marker = 'X' ,color='k')
        plt.ylabel('$\Delta$I (eV)', fontsize=12)
        plt.xlabel(r'$\gamma$ (Bohr$^{-1}$)', fontsize=12)
        plt.show()
