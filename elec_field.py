import numpy as np
from SparsePlace import SparsePlaceSpherical


class UniformField:
    def __init__(self, unit_vec=np.array([0,0,1])):
        self.unit_vec = unit_vec
    def eval_voltage(self, x, y, z):
        voltage = (self.unit_vec[0]*x+self.unit_vec[1]*y+self.unit_vec[2]*z)##  a field of 1 mV/mm
        return voltage ## mV

class ICMS:
    def __init__(self, x,y,z,conductivity):
        self.x, self.y, self.z, self.cond = x,y,z,conductivity
    def eval_voltage(self, x, y, z):
        r = np.sqrt((x-self.x)**2+(y-self.y)**2+(z-self.z)**2)*1e-03 ## converting mm to m
        voltage = (1e-06/(4*np.pi*self.cond*r)) ## converting microamp to amp, and then converting volts to millivolts         
        return voltage*1000 ## mV

class sparse_place_human:
    
    def __init__(self, J=None, r=1.1, multiplier=1, fname_save=None, fname_load=None, theta_patch=None, phi_patch=None):
        self.fname_save = fname_save 
        self.fname_load = fname_load
        if theta_patch is None or phi_patch is None:
            raise Exception('Define the coordinates of the electrodes')
        self.theta_patch, self.phi_patch = theta_patch, phi_patch
        self._define_sparse_place()
        r = r*10**(-1) ## mm to cm
        if J is not None:
            self.J = J/self.area_elec
        else:
            raise Exception('Define the Currents through the Electrodes')
        self.r = self.radius_head-r
        self._calc_voltage()
        self.multiplier = multiplier
        print("Electric Field Simulator Instantiated")

    def _define_sparse_place(self):
        
        radius_head = 9.2
        self.radius_head = radius_head
        thickness_scalp = 0.6
        thickness_skull = 0.5
        thickness_csf = 0.1 
        no_of_layers = 4
    
        skull_cond = 0.006
        brain_cond = 0.33
        csf_cond = 1.79
        scalp_cond = 0.3
    
        radius_vec_2layer = np.array([radius_head-thickness_skull-thickness_csf-thickness_scalp,radius_head-thickness_skull-thickness_scalp,radius_head-thickness_scalp, radius_head])
        cond_vec_2layer = np.array([brain_cond, csf_cond, skull_cond, scalp_cond])
    
        L = 600
    
        custom_grid_patch = True
        theta_patch =self.theta_patch
        phi_patch =self.phi_patch
    
        elec_radius_patch = 0.25
        elec_spacing = None 
        spacing = 0.02
    
        self.sparse_rodent = SparsePlaceSpherical(cond_vec=cond_vec_2layer, radius_vec=radius_vec_2layer, patch_size=None, elec_radius=elec_radius_patch, elec_spacing=elec_spacing, max_l=L, spacing=spacing, custom_grid=custom_grid_patch, theta_elec=theta_patch, phi_elec=phi_patch)
        self.area_elec = np.pi*(elec_radius_patch**2)*np.ones(len(self.theta_patch))
   
    def plot_patch(self,fname=None):
        self.sparse_rodent.plot_elec_pttrn(J=(len(self.theta_patch)+1)*self.area_elec, x_lim=[-10,10], y_lim=[-10,10], fname=fname)

    def plot_elec_pttrn(self,fname=None):
        self.sparse_rodent.plot_elec_pttrn(J=self.J, x_lim=[-10,10], y_lim=[-10,10], fname=fname)
    
    def plot_curr_density(self,fname=None):
        self.sparse_rodent.plot_curr_density(r=self.r, J=self.J, x_limit=[-2,2], y_limit=[-2,2], fname=fname, abs=False)
    
    def plot_voltage(self,fname=None):
        self.sparse_rodent.plot_voltage(r=self.r, J=self.J, x_limit=[-2,2], y_limit=[-2,2], fname=fname, abs=False)
    
    def _calc_curr_density(self):
        self.sparse_rodent.calc_curr_density(r=self.r, J=self.J)
    
    def _calc_Efield(self, x,y,z):
        x, y, z = x.reshape(-1,1)*10**(-1), y.reshape(-1,1)*10**(-1), z.reshape(-1,1)*10**(-1) ## mm->cm
        sph = self._cart_to_sph(np.hstack([x,y,z])) 
        r, theta, phi = sph[:,0].reshape(-1,1),sph[:,1].reshape(-1,1), sph[:,2].reshape(-1,1)
        return self.sparse_rodent.evaluate_Efield_fast(r,theta,phi, J=self.J) #mA/cm^2       

    def _calc_voltage(self):
        self.sparse_rodent.calc_voltage_diff_height(r_start=self.r, J=self.J, fname_load=self.fname_load, fname_save=self.fname_save)   
    
    def _cart_to_sph(self, pos):
        if len(pos.shape) == 1:
            pos = pos.reshape(1,-1)
        r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
        theta = np.arcsin(pos[:,2]/r.flatten()).reshape(-1,1)
        phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
        sph_pos = np.hstack([r,theta,phi]) 
        sph_pos[sph_pos[:,2]<0,2] = sph_pos[sph_pos[:,2]<0,2]+2*np.pi
        return sph_pos

    def eval_voltage(self, x,y,z):
        x, y, z = x.reshape(-1,1)*10**(-1), y.reshape(-1,1)*10**(-1), z.reshape(-1,1)*10**(-1) ## mm->cm
        sph = self._cart_to_sph(np.hstack([x,y,z])) 
        r, theta, phi = sph[:,0].reshape(-1,1),sph[:,1].reshape(-1,1), sph[:,2].reshape(-1,1)
        return self.multiplier*self.sparse_rodent.evaluate_voltage_fast(r,theta,phi)*10**3 #mV       


