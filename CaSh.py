import numpy as np
import re
import os
import h5py as h5

au_in_km = 1.496*10**8
t_in_s = 5.023*10**6 # time unit that sets G=1

class h5_Dump:
    """ Class for a single dump. """

    def compute_arrays(self):
        """ 
        Compute needed arrays. 
        
        input: 
        mass 
        position 
        velocity
        
        output: dependencies

        radius_spherical: position
        specific_potential_energy: mass radius_spherical
        specific_kinetic_energy: velocity
        semimajor_axis: mass specific_potential_energy specific_kinetic_energy
        eccentricity_vector: velocity position mass radius_spherical
        """
        
        with h5.File(self.dump_filename, "r") as h5_dump:
            sinks_n = 2
            sinks_key = 'sinks' #list(h5_dump.keys())[sinks_n]
            parts_n = 1
            parts_key = 'particles' #list(h5_dump.keys())[parts_n]
            header_n = 0
            header_key = 'header' #list(h5_dump.keys())[header_n]
            
            
            sinks_xyz_n = 7
            sinks_xyz_key = 'xyz' #list(h5_dump[sinks_key])[sinks_xyz_n]
            sinks_mass_n = 2
            sinks_mass_key = 'm' #list(h5_dump[sinks_key])[sinks_mass_n]
            sinks_vxyz_n = 6
            sinks_vxyz_key = 'vxyz' #list(h5_dump[sinks_key])[sinks_vxyz_n]

            
            print('Loading sinks properties (%s):' % sinks_key)
            print('    position (%s)' %  sinks_xyz_key)
            self.sinks_xyz = h5_dump[sinks_key][sinks_xyz_key][()]
            print('    mass (%s)' %  sinks_mass_key)
            self.sinks_mass = h5_dump[sinks_key][sinks_mass_key][()]
            print('    velocity (%s)' %  sinks_vxyz_key)
            self.sinks_vxyz = h5_dump[sinks_key][sinks_vxyz_key][()]

            parts_mass_n = 0
            parts_mass_key = 'massoftype' #list(h5_dump[parts_key])[parts_mass_n]
            parts_xyz_n = 8
            parts_xyz_key = 'xyz' #list(h5_dump[parts_key])[parts_xyz_n]
            parts_vxyz_n = 7
            parts_vxyz_key = 'vxyz' #list(h5_dump[parts_key])[parts_vxyz_n]

            print('Loading particles properties (%s):' % parts_key)
            print('    position (%s)' %  parts_xyz_key)
            self.parts_xyz = h5_dump[parts_key][parts_xyz_key][()]
            print('    mass (%s)' %  parts_mass_key)
            self.parts_mass = h5_dump[header_key][parts_mass_key][()][0]
            print('    velocity (%s)' %  parts_vxyz_key)
            self.parts_vxyz = h5_dump[parts_key][parts_vxyz_key][()]


        self.stellar_mass = np.sum(self.sinks_mass)

        self.cm_pos = np.sum(self.sinks_xyz*self.sinks_mass[:,None], axis=0)/np.sum(self.sinks_mass)
        self.cm_vel = np.sum(self.sinks_vxyz*self.sinks_mass[:,None], axis=0)/np.sum(self.sinks_mass)

        self.parts_xyz -= self.cm_pos
        self.parts_vxyz -= self.cm_vel

        
        self.parts_r = np.sqrt((self.parts_xyz*self.parts_xyz).sum(axis=1)) # particles spherical radius

        
        self.parts_specific_potential_energy = -self.stellar_mass/self.parts_r
        self.parts_specific_kinetic_energy = 0.5*(self.parts_vxyz*self.parts_vxyz).sum(axis=1)

        self.parts_semimajor_axis = -0.5*self.stellar_mass/(self.parts_specific_potential_energy+self.parts_specific_kinetic_energy)

        self.parts_l = np.cross(self.parts_xyz, self.parts_vxyz)
        self.parts_eccentricity_vector = np.cross(self.parts_vxyz, self.parts_l)/self.stellar_mass-self.parts_xyz/self.parts_r[:, None]#.repeat(3).reshape(-1,3)
        
        return

    def compute_cavity_semimajor_axis(self,
                                      n_bins=100,
                                      n_min=-1,
                                      n_max=-1):
        def find_cavity_edge(dens_profile): # Defined as the radius at which surface density first reaches half its maximum (Artymowicz&Lubow94)
            until_max = dens_profile[:np.argmax(dens_profile)+1]
            mask=until_max<(np.max(until_max)/2)
            cavity = until_max[mask]
    
            return len(cavity)

        if n_min < 0: n_min=np.min(self.parts_semimajor_axis[self.parts_semimajor_axis>0])
        if n_max < 0: n_max=np.max(self.parts_semimajor_axis)
        
        print(n_min, n_max, n_bins)
        bins=np.linspace(n_min, n_max, n_bins)

        masks=[]
        for i, ain in enumerate(bins[:-1]):
            aout=bins[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_semimajor_axis>=ain, 
                self.parts_semimajor_axis<aout))
            masks.append(cond)


        
        asd = np.array([len(x) for x in masks])#np.count_nonzero(self.a_masks, axis=1)
        
        self.cavity_semimajor_axis = bins[find_cavity_edge(asd)]
        #self.cavity_eccentricity = 5

        return

    
    def compute_cavity_orbital_parameters(self): # Improve how CaSh chooses particles to compute cavity orbital parameters with (filtering particles at a distance (compute_cavity_sma precision? one-sided?) from true cavity edge?)
        "Compute cavity orbital parameter"
        #sink_mass = np.sum(dump.snap.sinks['mass']).to('solar_mass').magnitude

        self.cav_idx = len(self.a_bins[self.a_bins<self.cavity_semimajor_axis])
        self.cav_a = self.a_bins[self.cav_idx]
        self.cav_e = np.sqrt(self.a_peric[self.cav_idx,2]**2+self.a_peric[self.cav_idx,1]**2+self.a_peric[self.cav_idx,0]**2)
        
        z_ax = np.array([0,0,1]) #line of sight
        lon_vec = np.cross(z_ax, self.a_h[self.cav_idx]) #line of ascending node vector
        x_ax = np.array([1,0,0]) #vernal equinox
        
        
        self.cav_omega = np.sign(self.a_peric[self.cav_idx,2])*np.arccos(lon_vec.dot(self.a_peric[self.cav_idx])/self.cav_e/np.sqrt(np.sum(lon_vec*lon_vec)))
        self.cav_i = np.arccos(self.a_h[self.cav_idx].dot(z_ax)/np.sqrt(np.sum(self.a_h[self.cav_idx]*self.a_h[self.cav_idx])))
        self.cav_Omega = np.sign(lon_vec[1])*np.arccos(lon_vec.dot(x_ax)/np.sqrt(np.sum(lon_vec*lon_vec)))

        return

    def compute_cavity_shape(self, points=100):
        "Compute cavity shape with Thiele-Innes elements"

        self.compute_cavity_orbital_parameters()

        omega, i, Omega = self.cav_omega, self.cav_i, self.cav_Omega #self.compute_cavity_orbital_parameters()
        
        E = np.linspace(0, 2*np.pi, points)
        
        P = np.array([np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.cos(i)*np.sin(Omega),
                      np.cos(omega)*np.sin(Omega)+np.sin(omega)*np.cos(i)*np.cos(Omega),
                      np.sin(omega)*np.sin(i)])
        
        Q = np.array([-np.sin(omega)*np.cos(Omega)-np.cos(omega)*np.cos(i)*np.sin(Omega),
                      -np.sin(omega)*np.sin(Omega)+np.cos(omega)*np.cos(i)*np.cos(Omega),
                      np.cos(omega)*np.sin(i)])
        
        A = np.cos(E)-self.cav_e
        
        B = np.sqrt(1-self.cav_e**2)*np.sin(E)
        
        orbit_xyz = self.cav_a*(np.outer(A,P)+np.outer(B,Q))#+self.cm_pos

        return orbit_xyz



    def compute_ring_average_e(self, r_in, r_out, n_bin):
        """ 
        Compute mean eccentricity in each bin between r_in and r_out.
        """
        
        
        self.a_bins=np.linspace(r_in, r_out, n_bin)

        self.a_masks=[]
        self.a_peric=np.zeros((len(self.a_bins)-1,3))
        self.a_h=np.zeros((len(self.a_bins)-1,3))
        
        for i, ain in enumerate(self.a_bins[:-1]):
            aout=self.a_bins[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_semimajor_axis>=ain, 
                self.parts_semimajor_axis<aout))
            self.a_masks.append(cond)
    
        for s, mask in enumerate(self.a_masks):
            #print(mask.T[0])
            if len(mask)>0:
                self.a_peric[s]=np.mean(self.parts_eccentricity_vector[mask], axis=0)
                self.a_h[s]=np.mean(self.parts_l[mask], axis=0)
            
        return 

    def compute_disc_orbital_parameters(self, r_in, r_out, n_bin):
        """ 
        Compute the average orbital parameters for each disc bin
        """
        
        self.anu_sma=np.linspace(r_in, r_out, n_bin)

        self.a_masks=[]
        self.a_peric=np.zeros((len(self.anu_sma)-1,3))
        self.a_h=np.zeros((len(self.anu_sma)-1,3))

        self.anu_ecc_vec=np.zeros((len(self.anu_sma)-1,3))
        self.anu_ecc_vec_std=np.zeros((len(self.anu_sma)-1,3))

        self.anu_spec_ang_mom=np.zeros((len(self.anu_sma)-1,3))

        z_ax=np.array([0,0,1])
        x_ax=np.array([1,0,0])
        
        for i, ain in enumerate(self.anu_sma[:-1]):
            aout=self.anu_sma[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_semimajor_axis>=ain, 
                self.parts_semimajor_axis<aout))
            self.a_masks.append(cond)
    
        for s, mask in enumerate(self.a_masks):
            if len(mask)>0:
                self.anu_spec_ang_mom[s] = np.mean(self.parts_l[mask], axis=0)
                self.anu_ecc_vec[s] = np.mean(self.parts_eccentricity_vector[mask], axis=0)
                self.anu_ecc_vec_std[s] = np.std(self.parts_eccentricity_vector[mask], axis=0)

        self.anu_ecc = np.sqrt(np.sum(self.anu_ecc_vec*self.anu_ecc_vec, axis=-1))
        self.anu_ecc_std = np.sqrt(np.sum(self.anu_ecc_vec**2*self.anu_ecc_vec_std**2, axis=-1))/self.anu_ecc

        self.anu_incl = np.arccos(self.anu_spec_ang_mom.dot(z_ax)
                                  /np.sqrt(np.sum(self.anu_spec_ang_mom**2, axis=-1)))

        self.anu_lon_vec = np.cross(z_ax, self.anu_spec_ang_mom)
        self.anu_Omega = np.arctan2(np.sqrt(np.sum(np.cross(x_ax,self.anu_lon_vec)**2, axis=-1))
                                    /np.sqrt(np.sum(self.anu_lon_vec**2, axis = -1)),
                                    self.anu_lon_vec.dot(x_ax)
                                    /np.sqrt(np.sum(self.anu_lon_vec**2, axis = -1)))

        
        self.anu_omega = np.arctan2(np.sqrt(np.sum(np.cross(self.anu_lon_vec, self.anu_ecc_vec)**2, axis=-1))
                                   /np.sqrt(np.sum(self.anu_ecc_vec**2, axis=-1))
                                   /np.sqrt(np.sum(self.anu_lon_vec**2, axis=-1)),
                           np.sum(self.anu_lon_vec*self.anu_ecc_vec, axis=-1)
                                   /np.sqrt(np.sum(self.anu_ecc_vec**2, axis=-1))
                                   /np.sqrt(np.sum(self.anu_lon_vec**2, axis=-1)))
            
        return 
    

    
    
    def __init__(self, filename):
        """ Load single dump. """

        self.dump_filename = filename #"FileNameMissing"

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
        
        self.parts_semimajor_axis = []
        self.parts_eccentricity_vector = []

        self.cavity_eccentricity = -1
        self.cavity_semimajor_axis = -1


        self.anu_sma = -1
        self.anu_ecc_vec = -1
        self.anu_ec_vec_std = -1
        self.anu_ecc = -1
        self.anu_ecc_std = -1
        self.anu_spec_ang_mom = -1
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


