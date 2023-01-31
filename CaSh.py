import numpy as np
import re
import os
import h5py as h5

au_in_km = 1.496*10**8
t_in_s = 5.023*10**6 # time unit that sets G=1

class h5_Dump:
    """ Class for a single dump. """

    def load_arrays(self, verbose=False):
        with h5.File(self.dump_filename, "r") as h5_dump:
            sinks_key = 'sinks' #list(h5_dump.keys())[sinks_n]
            parts_key = 'particles' #list(h5_dump.keys())[parts_n]
            header_key = 'header' #list(h5_dump.keys())[header_n]
            
            
            sinks_xyz_key = 'xyz' #list(h5_dump[sinks_key])[sinks_xyz_n]
            sinks_mass_key = 'm' #list(h5_dump[sinks_key])[sinks_mass_n]
            sinks_vxyz_key = 'vxyz' #list(h5_dump[sinks_key])[sinks_vxyz_n]

            
            if verbose: print('Loading sinks properties (%s):' % sinks_key)
            if verbose: print('    position (%s)' %  sinks_xyz_key)
            self.sinks_xyz = h5_dump[sinks_key][sinks_xyz_key][()]
            if verbose: print('    mass (%s)' %  sinks_mass_key)
            self.sinks_mass = h5_dump[sinks_key][sinks_mass_key][()]
            if verbose: print('    velocity (%s)' %  sinks_vxyz_key)
            self.sinks_vxyz = h5_dump[sinks_key][sinks_vxyz_key][()]

            parts_mass_key = 'massoftype' #list(h5_dump[parts_key])[parts_mass_n]
            parts_xyz_key = 'xyz' #list(h5_dump[parts_key])[parts_xyz_n]
            parts_vxyz_key = 'vxyz' #list(h5_dump[parts_key])[parts_vxyz_n]

            if verbose: print('Loading particles properties (%s):' % parts_key)
            if verbose: print('    position (%s)' %  parts_xyz_key)
            self.parts_xyz = h5_dump[parts_key][parts_xyz_key][()]
            if verbose: print('    mass (%s)' %  parts_mass_key)
            self.parts_mass = h5_dump[header_key][parts_mass_key][()][0]
            if verbose: print('    velocity (%s)' %  parts_vxyz_key)
            self.parts_vxyz = h5_dump[parts_key][parts_vxyz_key][()]

        if not self.arrays_loaded(): print("Found problems with loaded arrays.")
        elif self.arrays_loaded() and verbose: print("Arrays loaded correctly.")


        return

    def arrays_loaded(self, force_check=False):
        """ Check if arrays are correctly loaded. """

        if self.checked and not force_check: return True

        if len(self.sinks_mass)>0:
            if len(self.sinks_xyz)>0:
                if len(self.sinks_xyz)>0:
                    if self.parts_mass>0:
                        if len(self.parts_xyz)>0:
                            if len(self.parts_vxyz)>0:
                                if (len(self.sinks_mass) == len(self.sinks_xyz) and
                                    len(self.sinks_mass) == len(self.sinks_vxyz)):
                                   if (len(self.parts_xyz) == len(self.parts_vxyz)):
                                      self.checked = True
                                      return True
        
        return False

    def compute_arrays(self, force_comp = False):
        """ 
        Compute needed arrays for disc shape analysis. 
        
        It needs: 
        - sinks_mass: sinks masses
        - sinks_xyz: sinks positions
        - sinks_vxyz: sinks velocities
        
        - parts_mass: particles mass
        - parts_xyz: particles positions
        - parts_vxyz: particles velocities

        It computes:
        - stellar_mass: total stellar mass
        - cm_pos: center of mass position
        - cm_vel: center of mass velocity
        - parts_r: particles spherical radius
        - parts_sma: particles semi-major axis
        - parts_l: particles specific angular momentum
        - parts_ecc_vec: particles eccentricity vector
        """

        if self.arrays_computed and not force_comp: return # check the need to compute arrays

        if not self.arrays_loaded(): self.load_arrays() # load arrays if needed

        self.stellar_mass = np.sum(self.sinks_mass) # compute total stellar mass

        ## compute center of mass position and velocity
        self.cm_pos = np.sum(self.sinks_xyz*self.sinks_mass[:,None], axis=0)/np.sum(self.sinks_mass)
        self.cm_vel = np.sum(self.sinks_vxyz*self.sinks_mass[:,None], axis=0)/np.sum(self.sinks_mass)

        ## fix the center of mass in the origin
        self.parts_xyz -= self.cm_pos
        self.parts_vxyz -= self.cm_vel

        
        self.parts_r = np.sqrt((self.parts_xyz*self.parts_xyz).sum(axis=1)) # compute particles spherical radius

        ## compute particles semi-major axis
        self.parts_specific_potential_energy = -self.stellar_mass/self.parts_r
        self.parts_specific_kinetic_energy = 0.5*(self.parts_vxyz*self.parts_vxyz).sum(axis=1)
        self.parts_sma = -0.5*self.stellar_mass/(self.parts_specific_potential_energy+self.parts_specific_kinetic_energy)
        ## compute particles eccentricity vector
        self.parts_l = np.cross(self.parts_xyz, self.parts_vxyz)
        self.parts_ecc_vec = np.cross(self.parts_vxyz, self.parts_l)/self.stellar_mass-self.parts_xyz/self.parts_r[:, None]

        self.arrays_computed = True
        
        return

    def compute_cavity_sma(self, n_bins=-1, n_min=-1, n_max=-1):
        def find_cavity_edge(profile):
            """
            
            Find cavity edge defined as the radius at which surface
            density first reaches half its maximum (Artymowicz&Lubow94).
            
            """
            
            until_max = profile[:np.argmax(profile)+1]
            mask=until_max<(np.max(until_max)/2)
            cavity = until_max[mask]
    
            return len(cavity)

        ## check for function arguments
        if n_bins < 0: n_bins=100
        if n_min < 0: n_min=np.min(self.parts_sma[self.parts_sma>0])
        if n_max < 0: n_max=np.max(self.parts_sma)

        ## bin particles in smi-major axis
        bins=np.linspace(n_min, n_max, n_bins)

        masks=[]
        for i, ain in enumerate(bins[:-1]):
            aout=bins[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_sma>=ain, 
                self.parts_sma<aout))
            masks.append(cond)

        ## find cavity semi-major axis
        self.cavity_sma = bins[find_cavity_edge(np.array([len(x) for x in masks]))]
        
        return

    
    def compute_cavity_orbital_parameters(self):
        """ Compute cavity orbital parameter """
        # Improve how CaSh chooses particles to compute cavity orbital parameters with (filtering particles at a distance (compute_cavity_sma precision? one-sided?) from true cavity edge?)
        
        self.cav_idx = len(self.anu_sma[self.anu_sma<self.cavity_sma])
        self.cav_a = self.anu_sma[self.cav_idx]
                
        self.cav_e, self.cav_i, self.cav_Omega, self.cav_omega = self.compute_orbital_parameters(self.anu_ecc_vec[self.cav_idx], self.anu_l[self.cav_idx])
                
        return

    def compute_cavity_shape(self, points=100):
        "Compute cavity shape with Thiele-Innes elements"

        self.compute_cavity_orbital_parameters()

        orbit_xyz = self.compute_orbit_shape(np.array([self.cav_a]),
                                             np.array([self.cav_e]),
                                             np.array([self.cav_i]),
                                             np.array([self.cav_Omega]),
                                             np.array([self.cav_omega])) 

        return orbit_xyz[0]

    def compute_disc_shape(self, points=100):
        "Compute disc shape with Thiele-Innes elements"

        self.compute_disc_orbital_parameters()

        orbit_xyz = self.compute_orbit_shape(self.anu_sma[:-1], self.anu_ecc, self.anu_incl, self.anu_Omega, self.anu_omega)
        return orbit_xyz


    def compute_orbit_shape(self, a, e, i, Omega, omega, points=100):
        "Compute cavity shape with Thiele-Innes elements"

        E = np.linspace(0, 2*np.pi, points)[None,:]

        P = np.array([np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.cos(i)*np.sin(Omega),
                      np.cos(omega)*np.sin(Omega)+np.sin(omega)*np.cos(i)*np.cos(Omega),
                      np.sin(omega)*np.sin(i)]).T
        Q = np.array([-np.sin(omega)*np.cos(Omega)-np.cos(omega)*np.cos(i)*np.sin(Omega),
                      -np.sin(omega)*np.sin(Omega)+np.cos(omega)*np.cos(i)*np.cos(Omega),
                      np.cos(omega)*np.sin(i)]).T

        A = (np.cos(E).T-e).T
        B = (np.sqrt(1-e**2)*np.sin(E).T).T
        
        orbit_xyz = a[:,None,None]*(A[...,None]*P[:,None,:]+B[...,None]*Q[:,None,:])

        return orbit_xyz

    
    
    def compute_disc_orbital_parameters(self, r_in=10, r_out=100, n_bin=10, force_comp=False):
        """ 
        Compute the average orbital parameters for each disc bin
        """

        if not self.arrays_computed or force_comp: self.compute_arrays() # load arrays if needed
        
        self.anu_sma=np.linspace(r_in, r_out, n_bin)

        self.a_masks=[]

        self.anu_ecc_vec=np.zeros((len(self.anu_sma)-1,3))
        self.anu_ecc_vec_std=np.zeros((len(self.anu_sma)-1,3))

        self.anu_l=np.zeros((len(self.anu_sma)-1,3))

        z_ax=np.array([0,0,1])
        x_ax=np.array([1,0,0])
        
        for i, ain in enumerate(self.anu_sma[:-1]):
            aout=self.anu_sma[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_sma>=ain, 
                self.parts_sma<aout))
            self.a_masks.append(cond)
    
        for s, mask in enumerate(self.a_masks):
            if len(mask)>0:
                self.anu_l[s] = np.mean(self.parts_l[mask], axis=0)
                self.anu_ecc_vec[s] = np.mean(self.parts_ecc_vec[mask], axis=0)
                self.anu_ecc_vec_std[s] = np.std(self.parts_ecc_vec[mask], axis=0)

        self.anu_ecc, self.anu_incl, self.anu_Omega, self.anu_omega = self.compute_orbital_parameters(self.anu_ecc_vec, self.anu_l)
        
        return


    def compute_orbital_parameters(self, ecc_vec, l):
        """ 
        Compute (list of) orbital parameters given (list of) specific angular momentum, eccentricity vector.

        input:
            ecc_vec, shape (:, 3)
            l, shape (:, 3)
        
        """

        z_ax=np.array([0,0,1])
        x_ax=np.array([1,0,0])
        
        ecc = np.sqrt(np.sum(ecc_vec*ecc_vec, axis=-1))
        ##ecc_std = np.sqrt(np.sum(ecc_vec**2*ecc_vec_std**2, axis=-1))/ecc

        incl = np.arccos(l.dot(z_ax)
                         /np.sqrt(np.sum(l**2, axis=-1)))

        lon_vec = np.cross(z_ax, l)
        Omega = np.arctan2(np.sign(np.take(lon_vec,1,-1))*np.sqrt(np.sum(np.cross(x_ax,lon_vec)**2, axis=-1))
                           /np.sqrt(np.sum(lon_vec**2, axis = -1)),
                           lon_vec.dot(x_ax)
                           /np.sqrt(np.sum(lon_vec**2, axis = -1)))

        
        omega = np.arctan2(np.sign(np.take(ecc_vec,2,-1))*np.sqrt(np.sum(np.cross(lon_vec, ecc_vec)**2, axis=-1))
                                   /np.sqrt(np.sum(ecc_vec**2, axis=-1))
                                   /np.sqrt(np.sum(lon_vec**2, axis=-1)),
                           np.sum(lon_vec*ecc_vec, axis=-1)
                                   /np.sqrt(np.sum(ecc_vec**2, axis=-1))
                                   /np.sqrt(np.sum(lon_vec**2, axis=-1)))

        #return ecc, ecc_std, incl, Omega, omega
        return ecc, incl, Omega, omega


    

    
    
    def __init__(self, filename):
        """ Load single dump. """

        self.dump_filename = filename #"FileNameMissing"

        self.checked = False
        self.arrays_computed = False

        self.gas_particles = []
        self.dust_particles = []
        self.sink_particles = []

        self.time = -1
        self.gas = -1
        self.sinks = -1
        self.accr = -1
        self.stellar_mass = -1

        self.sinks_mass = []    
        self.parts_mass = []
        self.parts_xyz = []
        self.parts_vxyz = []

        #        self.cm_pos = []
        #        self.cm_vel = []
    
        self.parts_r = []
    
        self.parts_specific_potential_energy = []
        self.parts_specific_kinetic_energy = []

        self.parts_l = []
        
        self.parts_sma = []
        self.parts_ecc_vec = []

        self.cavity_eccentricity = -1
        self.cavity_sma = -1


        self.anu_sma = -1
        self.anu_ecc_vec = -1
        self.anu_ec_vec_std = -1
        self.anu_ecc = -1
        self.anu_ecc_std = -1
        self.anu_l = -1
        self.anu_incl = -1
        self.anu_lon_vec = -1
        self.anu_Omega = -1
        self.anu_omega = -1

        self.cav_Omega = -1
        self.cav_omega = -1
        self.cav_i = -1
        self.cav_e = -1
        self.cav_a = -1

        self.cav_idx = -1
                
class Simulation:
    """ Class for a set of dumps. """

    def __init__(self, simname = "GGTau3", start = 0, end = -1, skip = 1):
        """ Load simulation dumps. """

        self.dumps = []

        self.simname = simname
        self.start = start
        self.end = end
        self.skip = skip

        patt=re.compile(simname+'_[0-9]{5}.ascii')
    
        #Loading full dumps
    
        fulldumps = [dump for dump in sorted(os.listdir('.')) if patt.match(dump)]
    
        if end > 0: chosendumps=fulldumps[start:end:skip]
        else: chosendumps=fulldumps[start::skip]
    
        print("Found",len(fulldumps), "full dumps. Selected", len(chosendumps),"dumps.")
        print(chosendumps)

        for dump in chosendumps:
            self.dumps.append(Dump(dump))


