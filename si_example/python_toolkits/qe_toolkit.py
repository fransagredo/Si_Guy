import xml.dom.minidom as xml
import xml.etree.ElementTree as et
import os, re, shutil, subprocess, numpy as np
import wannier_tuning_toolkit as wt
import srsh_correction
import matplotlib.pyplot as plt

har_to_eV = 27.21138
ry_to_eV = 13.6057
madelung = 2.8373
hartree_to_ry = 2
ang_to_bohr = 1.8897
pw_path = '/home1/07352/tg866516/apps/quantum_espresso/qe_6.2_wannier_constr_dat/q-e_private/bin/pw.x'

def get_lumo_homo(xml_path, ik):
    '''
    This function returns the lumo and homo eigenvalues at kpoint ikp
    ikp run from 1 to number of kpoints
    '''

    if not os.path.isfile(xml_path):
        return None, None

    tree = et.parse(xml_path)
    root = tree.getroot()

    # get all kpoints
    kpoints = root.findall('./output/band_structure/ks_energies')

    # get the ik kpoint
    kp = kpoints[ik-1]

    # get a list of this kpoint eigenvalues
    eig = list(map(float, kp[2].text.split()))

    # get a list of this kpoint occupations
    occ = list(map(float, kp[3].text.split()))

    for i in range(0, len(occ)):
        if occ[i] < 0.98:
            if occ[i] > 0.01:
                print('you might have partial occupations')
            break;

    lumo = eig[i] * har_to_eV
    homo = eig[i-1] * har_to_eV

    return lumo, homo

def get_toten(xml_path):
    '''
    Get total energy in eV
    '''
    if not os.path.isfile(xml_path):
        return None

    tree = et.parse(xml_path)
    root = tree.getroot()

    toten_har = root.find('./output/total_energy/etot')
    toten_har = float(toten_har.text)
    toten_ev = toten_har * har_to_eV

    return toten_ev
    
def get_gap(xml_path, ik):
    '''
    This function returns the lumo-homo at kpoint ikp
    ikp run from 1 to number of kpoints
    specify ikp=-1 for the minimal gap in the band structure
    '''
    tree = et.parse(xml_path)
    root = tree.getroot()

    # get all kpoints
    kpoints = root.findall('./output/band_structure/ks_energies')
    nkp = len(kpoints)

    gaps = []
    lumos = []
    homos = []

    for kp in range(1,nkp):
        [lumo, homo] = get_lumo_homo(xml_path, kp)
        lumos.append(lumo)
        homos.append(homo)
        gaps.append(lumo - homo)

    if ik == -1:
        gap = min(lumos)-max(homos)
    else:
        gap = gaps[ik-1]


    return gap

def num_in_dir_name(dir_name):
    '''
    Get the first number in dir name
    '''

    match = re.findall(r"(?:-?\d+.?\d+|-?\d)", dir_name)
    num = float(match[0])

    return num
    
def get_gaps(seedname, ik):
    '''
    Print gaps from all directories in current directory
    '''
    all_dirs = []
    curr_dir = os.getcwd()
    for subdir, dirs, files in os.walk(curr_dir):
        for d in dirs:
            xml_path = curr_dir+'/'+d+'/calculations/'+seedname+'.xml'
            if os.path.isfile(xml_path):
                this_dir = {}
                this_dir['dir_name'] = d
                this_dir['val'] = num_in_dir_name(d)
                this_dir['gap'] = get_gap(xml_path, ik)
                all_dirs.append(this_dir)

    all_dirs_sorted = sorted(all_dirs, key = lambda i: i['val']) #sort the list of dictionaries by the 'val' key

    for j in all_dirs_sorted:
        print(j['val'])
        print(j['gap'])
        print()

    if len(all_dirs_sorted) == 2:
        return abs(all_dirs_sorted[1]['gap']-all_dirs_sorted[0]['gap'])

def get_totens(seedname):
    '''
    Print total energy from all directories in current directory
    '''
    all_dirs = []
    curr_dir = os.getcwd()
    for subdir, dirs, files in os.walk(curr_dir):
        for d in dirs:
            xml_path = curr_dir+'/'+d+'/calculations/'+seedname+'.xml'
            if os.path.isfile(xml_path):
                this_dir = {}
                this_dir['dir_name'] = d
                this_dir['val'] = num_in_dir_name(d)
                this_dir['toten'] = get_toten(xml_path)
                all_dirs.append(this_dir)

    all_dirs_sorted = sorted(all_dirs, key = lambda i: i['val']) #sort the list of dictionaries by the 'val' key

    for j in all_dirs_sorted:
        print(j['val'])
        print(j['toten'])
        print()

def remove_save_dirs():
    '''
    Remove all save dirs in current dir and subdirs
    '''
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(curr_dir):
        for d in dirs:
            if d.endswith('.save'):
                shutil.rmtree(os.path.join(root, d))

def access_time():
    '''
    Change access time of all directories and files in current directory
    '''
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(curr_dir):
        for d in dirs:
            subprocess.call(["touch","-a",os.path.join(root, d)])
        for f in files:
            subprocess.call(["touch","-a",os.path.join(root, f)])

def ecut_kp_dirs(input_files_dir, seedname, val_min, val_max, delta, is_ecut = True):
    '''
    Specify if you converge ecut (default) or kpoints
    Copy the files in input_files_dir and create dirs with name of the value of ecut or kpoints
    replace the value of ecut or kpints in each dir
    make sure ecut is defined 'ecutwfc = xxx' or kpoints is 'x x x 0 0 0'
    '''
    if is_ecut:
        for ecut in np.arange(val_min, val_max, delta):
            shutil.copytree(input_files_dir, str(ecut))
            in_file_path = str(ecut)+'/'+seedname+'.in'
            if os.path.isfile(in_file_path):
                with open(in_file_path) as inp:
                    new_inp = inp.read().replace('ecutwfc = xxx' ,'ecutwfc = '+str(ecut))
                with open(in_file_path, "w") as inp:
                    inp.write(new_inp)
    
    else:
        for kp in np.arange(val_min, val_max, delta):
            kp_dir = str(kp)*3
            shutil.copytree(input_files_dir, kp_dir)
            in_file_path = kp_dir+'/'+seedname+'.in'
            if os.path.isfile(in_file_path):
                with open(in_file_path) as inp:
                    new_inp = inp.read().replace('x x x 0 0 0' ,str(kp)+' '+str(kp)+' '+str(kp)+' 0 0 0')
                with open(in_file_path, "w") as inp:
                    inp.write(new_inp)

def ecut_kp_submit(seedname,val_min, val_max, delta, nb, nk):
    '''
    Run pw.x in all dirs with names from val_min to val_max in jumps of delta
    Define nb and nk flags
    '''
    for val in np.arange(val_min, val_max, delta):
        dir_name = str(val)
        os.chdir(dir_name)
        with open(seedname+".out", "w") as qe_out:
            subprocess.call(["ibrun",pw_path,"-nb",str(nb),"-nk",str(nk),"-input",seedname+".in"], stdout = qe_out)
        os.chdir('..')
        
def ecut_dirs_koopmans(input_files_dir, seedname, val_min, val_max, delta):
    '''
    similar to previous function, only for dirs of converging KC
    i.e. the name of the dir is ecut, and it has two subdirs: N and N-1
    '''
    for ecut in np.arange(val_min, val_max, delta):
        shutil.copytree(input_files_dir, str(ecut))
        in_file_path_N = str(ecut)+'/N_/'+seedname+'.in'
        in_file_path_Nm1 = str(ecut)+'/N-1_/'+seedname+'.in'

        if os.path.isfile(in_file_path_N):
            with open(in_file_path_N) as inp:
                new_inp = inp.read().replace('ecutwfc = xxx' ,'ecutwfc = '+str(ecut))
            with open(in_file_path_N, "w") as inp:
                inp.write(new_inp)

        if os.path.isfile(in_file_path_Nm1):
            with open(in_file_path_Nm1) as inp:
                new_inp = inp.read().replace('ecutwfc = xxx' ,'ecutwfc = '+str(ecut))
            with open(in_file_path_Nm1, "w") as inp:
                inp.write(new_inp)

def calc_kc_gap(dir_path, seedname):
    '''
    Calculate koopmans compliance and gap at gamma for a single directory with 'N_' and 'N-1_' subdirs
    '''

    found_N_dir = False
    found_Nm1_dir = False

    # search for the N and N-1 directories
    for root, dirs, files in os.walk(dir_path):
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
        return None, None

    xml_N = dir_path+'/'+N_dir+'/calculations/'+seedname+'.xml'
    xml_Nm1 = dir_path+'/'+Nm1_dir+'/calculations/'+seedname+'.xml'
    E_N = get_toten(xml_N)
    E_Nm1 = get_toten(xml_Nm1)

    [gamma_bohr_N, alpha_N, onebyeps_N, L_ang_N, lambda_ry_N] = wt.get_input_srsh(dir_path+'/'+N_dir+'/'+seedname+'.in')

    ICC = srsh_correction.srsh_E_corr(1, alpha_N, gamma_bohr_N * ang_to_bohr, onebyeps_N, L_ang_N)
    
    [lumo, homo] = get_lumo_homo(xml_N, 1)
    
    if E_N is None or E_Nm1 is None or homo is None or lumo is None:
        return None, None

    I = (E_Nm1 + ICC - E_N) + homo
    gap = lumo-homo

    return I,gap

def get_kc_gap(seedname, printed = 'gap'):
    '''
    Print koopmans compliance or gap at gamma point (specify which one: printed = 'gap' or 'kc') from all directories in current directory
    '''
    
    all_dirs = []
    curr_dir = os.getcwd()
    for subdir, dirs, files in os.walk(curr_dir):
        for d in dirs:
            dir_path = curr_dir+'/'+d
            [I,gap] = calc_kc_gap(dir_path, seedname)
            if I is not None:
               this_dir = {}
               this_dir['dir_name'] = d
               this_dir['val'] = num_in_dir_name(d)
               this_dir['KC'] = I
               this_dir['gap'] = gap
               all_dirs.append(this_dir)

    #remove duplicates from dictionary
    all_dirs = [i for n, i in enumerate(all_dirs) if i not in all_dirs[n + 1:]]
        
    #sort the list of dictionaries by the 'val' key
    all_dirs_sorted = sorted(all_dirs, key = lambda i: i['val'])

    # print
    for j in all_dirs_sorted:
        print(j['val'])
        if printed == 'gap':
            print(j['gap'])
        else:
            print(j['KC'])
        print()

def indirect_gap(dat_path, num_occ_bands, num_points_per_band):
    '''
    Calculate the indirect gap from a wannier interpolated band structure.
    Give the path to the dat file. Give the number of occupied bands in this band structure, i.e. how many of the wannierized bands are occupied.
    Give the number of sampled points per band, i.e. how many points there are between blank line in the dat file
    '''

    e = np.genfromtxt(dat_path, dtype = np.float64, usecols = 1)
    
    tot_occ = num_occ_bands*num_points_per_band
    occ = e[0:tot_occ]
    unocc = e[tot_occ:]
    
    homo = max(occ)
    lumo = min(unocc)

    gap = lumo-homo

    return gap

def get_dipole_and_volume(out_file_path):
    '''
    This function returns from lelfield calculation
    - Electronic Dipole per cell (Ry a.u.)
    - unit-cell volume in (a.u.)^3
    '''

    with open(out_file_path, 'r', errors='replace') as out:
        lines = out.readlines()

    # finding dipole
    for line in reversed(lines):
        match_dipole = re.search(r'Electronic Dipole per cell', line)
        if match_dipole:
            match = re.search(r'-?\d+.?\d+E?-?\d+', line)
            dipole_string = match.group()
            break

    # finding u.c. volume
    for line in lines:
        match_vol = re.search(r'unit-cell volume          =', line)
        if match_vol:
            match = re.search(r'-?\d+.?\d+', line)
            vol_string = match.group()
            break

    dipole = float(dipole_string)
    vol = float(vol_string)

    return dipole,vol

def calc_eps(dir_path, seedname):
    '''
    Calculate epsilon_infinity for a dir with two subdirs: 'f0' and 'f1'
    Assumes electric field in 3rd direction of 0.001
    '''

    found_f0_dir = False
    found_f1_dir = False

    # search for the 0 and 0.001 directories
    for root, dirs, files in os.walk(dir_path):
        for dir0 in dirs:
            dir_name = os.path.join(dir0)
            match = re.search(r"(?:f0|f1)", dir_name)
            if match:
                if match[0] == 'f0':
                    f0_dir = dir_name
                    found_f0_dir = True
                elif match[0] == 'f1':
                    f1_dir = dir_name
                    found_f1_dir = True
    
    if not found_f0_dir or not found_f1_dir:
        return None

    # pulling relevant values
    [D_f0, vol_f0] = get_dipole_and_volume(dir_path+'/'+f0_dir+'/'+seedname+'.out')
    [D_f1, vol_f1] = get_dipole_and_volume(dir_path+'/'+f1_dir+'/'+seedname+'.out')

    # sanity check of volume
    if vol_f0 != vol_f1:
        print('two different volumes!')

    # epsilon
    epsilon_infinity = 4*np.pi/vol_f0*(D_f1-D_f0)/0.001+1

    return epsilon_infinity

def get_eps(seedname, plot = False):
    '''
    Print epsilon_infinity from all directories in current directory
    '''
    
    all_dirs = []
    curr_dir = os.getcwd()
    for subdir, dirs, files in os.walk(curr_dir):
        for d in dirs:
            dir_path = curr_dir+'/'+d
            eps = calc_eps(dir_path, seedname)
            if eps is not None:
               this_dir = {}
               this_dir['dir_name'] = d
               this_dir['val'] = num_in_dir_name(d)
               this_dir['eps'] = eps
               all_dirs.append(this_dir)

    #remove duplicates from dictionary
    all_dirs = [i for n, i in enumerate(all_dirs) if i not in all_dirs[n + 1:]]
        
    #sort the list of dictionaries by the 'val' key
    all_dirs_sorted = sorted(all_dirs, key = lambda i: i['val'])

    vals = []
    epsilons = []
    # print
    for j in all_dirs_sorted:
        print(j['val'])
        print(j['eps'])
        print()
        vals.append(j['val'])
        epsilons.append(j['eps'])

    # plot
    if plot:
        plt.plot(vals, epsilons, color='k')
        plt.scatter(vals, epsilons, marker = 'X' ,color='k')
        plt.show()
