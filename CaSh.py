import numpy as np
import re
import os
import plonk

au_in_km = 1.496*10**8
t_in_s = 5.023*10**6 # time unit that sets G=1

class Dump:
    """ Class for a single dump. """

    def compute_arrays(self, dd = 5):
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
        
        self.sinks_mass = np.array([part[3] for part in self.sink_particles])
        
        self.stellar_mass = np.sum(self.sinks_mass)
        
        self.parts_mass = np.array([part[3] for part in self.gas_particles])
        self.parts_xyz = np.array([part[0:3] for part in self.gas_particles])
        self.parts_vxyz = np.array([part[6+dd:9+dd] for part in self.gas_particles])/au_in_km*t_in_s # velocities are in km/s
        
        self.parts_r = np.sqrt((self.parts_xyz*self.parts_xyz).sum(axis=1)) # particles spherical radius
        
        self.parts_specific_potential_energy = -self.stellar_mass/self.parts_r
        self.parts_specific_kinetic_energy = 0.5*(self.parts_vxyz*self.parts_vxyz).sum(axis=1)
        
        
        self.parts_semimajor_axis = -0.5*self.stellar_mass/(self.parts_specific_potential_energy+self.parts_specific_kinetic_energy)
        self.parts_eccentricity_vector = np.cross(self.parts_vxyz, np.cross(self.parts_xyz, self.parts_vxyz))/self.stellar_mass-self.parts_xyz/self.parts_r.repeat(3).reshape(-1,3)
        
        return


    def compute_cavity_semimajor_axis(self, slices=8):

        def find_cavity_edge(dens_profile):
            until_max = dens_profile[:np.argmax(dens_profile)+1]
            mask=until_max<(np.max(until_max)/2)
            cavity = until_max[mask]
    
            return len(cavity)


        snap=plonk.load_snap(self.dump_filename.split('.')[0]+".h5")

        profiles=[]

        
        dtheta=2*np.pi/slices
        
        for j in range(0,slices):
            min_ang = dtheta*j
            max_ang = dtheta*(j+1)
            sign = 1
    
            if min_ang > np.pi: min_ang -= 2*np.pi
            if max_ang > np.pi: max_ang -= 2*np.pi
            mask = np.arctan2(snap['y'], snap['x']) > np.min([min_ang, max_ang])
            subsnap = snap[mask]
            mask = np.arctan2(subsnap['y'], subsnap['x']) < np.max([min_ang, max_ang])
            subsnap = subsnap[mask]
    
            profiles.append(plonk.load_profile(subsnap))

        radius = []
        index = []
    
        for profile in profiles:
            idx = find_cavity_edge(profile['surface_density'])
            index.append(idx)
            radius.append(profile['radius'][idx].to('au').magnitude)
    
        max_rad = np.max(radius)
        min_rad = np.min(radius)

        self.cavity_semimajor_axis = (max_rad+min_rad)/2
        self.cavity_eccentricity = (max_rad/min_rad-1)/(1+max_rad/min_rad)

        return


    def compute_ring_average_e(self, r_in, r_out, n_bin):
        """ 
        Compute mean eccentricity in each bin between r_in and r_out.
        """
        
        
        self.a_bins=np.linspace(r_in, r_out, n_bin)

        self.a_masks=[]
        self.a_peric=np.zeros((len(self.a_bins)-1,3))
        
        for i, ain in enumerate(self.a_bins[:-1]):
            aout=self.a_bins[i+1]
            cond=np.argwhere(np.logical_and(
                self.parts_semimajor_axis>=ain, 
                self.parts_semimajor_axis<aout))
            self.a_masks.append(cond)
    
        for s, mask in enumerate(self.a_masks):
            if len(mask.T[0])>0:
                self.a_peric[s]=np.mean(self.parts_eccentricity_vector[mask.T[0]], axis=0)
            
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
    
        self.parts_r = []
    
        self.parts_specific_potential_energy = []
        self.parts_specific_kinetic_energy = []
        
        self.parts_semimajor_axis = []
        self.parts_eccentricity_vector = []

        self.cavity_eccentricity = -1
        self.cavity_semimajor_axis = -1
    
        with open(filename) as dump:
            dump_particles = dump.readlines()

        # x y z mass h density vx vy vz divv dt itype

        for dump_line, dump_particle in enumerate(dump_particles):
            if dump_particle[0] == '#':
                if dump_particle.startswith("# time:"):
                    self.time = float(dump_particles[dump_line+1].split()[1])
                elif dump_particle.startswith("# npart:"):
                    self.gas = int(dump_particles[dump_line+1].split()[1])
                    self.sinks = int(dump_particles[dump_line+1].split()[3])
                    self.accr = int(dump_particles[dump_line+1].split()[24])
            else:
                itype = dump_particle.split()[-1]

                if itype == '1':
                    self.gas_particles.append([float(strnum) for strnum in dump_particle.split()[:-1]]) #map(float,dump_particle.split()[:-1]))
                elif itype == '3':
                    self.sink_particles.append([float(strnum) for strnum in dump_particle.split()[:-1]])
                elif itype == '2':
                    self.dust_particle.append([float(strnum) for strnum in dump_particle.split()[:-1]])
                else:
                    pass

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


