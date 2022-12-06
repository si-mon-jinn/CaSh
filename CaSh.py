import numpy as np
import re
import os
import plonk

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
        
        self.sinks_mass = self.snap.sinks['mass'].to('solar_mass').magnitude
        
        self.stellar_mass = np.sum(self.sinks_mass)

        self.cm_pos = np.sum(self.snap.sinks['position'].to('au').magnitude*self.sinks_mass[:,np.newaxis], axis=0)/np.sum(self.sinks_mass)
        self.cm_vel = np.sum(self.snap.sinks['velocity'].to('km/s').magnitude*self.sinks_mass[:,np.newaxis], axis=0)/np.sum(self.sinks_mass) /au_in_km*t_in_s

        
        self.parts_mass = self.snap['mass'].to('solar_mass').magnitude
        self.parts_xyz = self.snap['position'].to('au').magnitude - self.cm_pos
        self.parts_vxyz = self.snap['velocity'].to('km/s').magnitude/au_in_km*t_in_s - self.cm_vel

        
        self.parts_r = np.sqrt((self.parts_xyz*self.parts_xyz).sum(axis=1)) # particles spherical radius
        
        self.parts_specific_potential_energy = -self.stellar_mass/self.parts_r
        self.parts_specific_kinetic_energy = 0.5*(self.parts_vxyz*self.parts_vxyz).sum(axis=1)
        
        
        self.parts_semimajor_axis = -0.5*self.stellar_mass/(self.parts_specific_potential_energy+self.parts_specific_kinetic_energy)

        self.parts_l = np.cross(self.parts_xyz, self.parts_vxyz)
        self.parts_eccentricity_vector = np.cross(self.parts_vxyz, self.parts_l)/self.stellar_mass-self.parts_xyz/self.parts_r.repeat(3).reshape(-1,3)
        
        return


    def compute_cavity_semimajor_axis(self, slices=8):

        def find_cavity_edge(dens_profile): # Defined as the radius at which surface density first reaches half its maximum (Artymowicz&Lubow94)
            until_max = dens_profile[:np.argmax(dens_profile)+1]
            mask=until_max<(np.max(until_max)/2)
            cavity = until_max[mask]
    
            return len(cavity)

        Omega, i = 0, 0
        
        if True:
            h_parts = np.mean(self.parts_l, axis=0) # particles mean angular momentum
            i = np.arccos(np.dot([0,0,1], h_parts)/np.sqrt(np.sum(h_parts**2))) # disc inclination wrt z axis
                
            n_parts = np.cross([0,0,1], h_parts) # particles mean line of nodes
            Omega = np.arccos(np.dot([0,1,0],n_parts)/np.sqrt(np.sum(n_parts**2))) # particles mean longitude of the scending node

        # Rotate by Omega around z axis
        yy = self.parts_xyz[:,0]*np.sin(-Omega)+self.parts_xyz[:,1]*np.cos(-Omega)
        xx = self.parts_xyz[:,0]*np.cos(-Omega)-self.parts_xyz[:,1]*np.sin(-Omega)

        # Rotate by i around the y axis
        zz = -self.parts_xyz[:,0]*np.sin(i)+self.parts_xyz[:,2]*np.cos(i)
        xx = self.parts_xyz[:,0]*np.cos(i)+self.parts_xyz[:,2]*np.sin(i)

        # Shift each particle so that the system center of mass is in (0,0,0)
        self.snap['x']=self.snap['x'].to('au')-self.snap['x'][0].to('au')/self.snap['x'][0].to('au').magnitude*self.cm_pos[0]
        self.snap['y']=self.snap['y'].to('au')-self.snap['y'][0].to('au')/self.snap['y'][0].to('au').magnitude*self.cm_pos[1]
        self.snap['z']=self.snap['z'].to('au')-self.snap['z'][0].to('au')/self.snap['z'][0].to('au').magnitude*self.cm_pos[2]

        # Isolate a slice of disc, and store the density profile
        
        profiles=[]
        
        dtheta=2*np.pi/slices
        
        for j in range(0,slices):
            min_ang = dtheta*j
            max_ang = dtheta*(j+1)
            sign = 1
    
            if min_ang > np.pi: min_ang -= 2*np.pi
            if max_ang > np.pi: max_ang -= 2*np.pi
            mask1 = np.arctan2(yy, xx) > np.min([min_ang, max_ang])
            subsnap = self.snap[mask1]
            subxx = xx[mask1]
            subyy = yy[mask1]
            mask2 = np.arctan2(subyy, subxx) < np.max([min_ang, max_ang])
            subsnap = subsnap[mask2]

            profiles.append(plonk.load_profile(subsnap))

        radius = []
        index = []
    
        for j,profile in enumerate(profiles):
            idx = find_cavity_edge(profile['surface_density'])
            index.append(idx)
            radius.append(profile['radius'][idx].to('au').magnitude)
            
        max_rad = np.max(radius)
        min_rad = np.min(radius)

        self.cavity_semimajor_axis = (max_rad+min_rad)/2
        self.cavity_eccentricity = (max_rad/min_rad-1)/(1+max_rad/min_rad)

        return
    
    def compute_cavity_orbital_parameters(self):
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
            if len(mask.T[0])>0:
                self.a_peric[s]=np.mean(self.parts_eccentricity_vector[mask.T[0]], axis=0)
                self.a_h[s]=np.mean(self.parts_l[mask.T[0]], axis=0)
            
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

        self.snap=plonk.load_snap(self.dump_filename.split('.')[0]+".h5")

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


