import numpy as np
import time
from ElectricGrid import PreDefElecGrid
from TransferFunction import TransferFunction
import pyshtools as shp_harm
import matplotlib.pyplot as plt
import seaborn as sns
import ray

class SparsePlaceSpherical(PreDefElecGrid, TransferFunction):

    def __init__(self, cond_vec, radius_vec, patch_size, elec_radius, elec_spacing, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.custom_grid = custom_grid
        if self.custom_grid is False:
            self.patch_size = patch_size * 10**(-2)
            self.elec_spacing = elec_spacing * 10**(-2)
        else:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec
        self.elec_radius = elec_radius * 10**(-2)
        self.curr_density_calculated = False
    
    def evaluate_voltage_fast(self,r,theta,phi):
        if self.voltage_diff_height_flag is False:
            raise Exception('Calculate the current density first using calc_curr_density() function')
        r, theta, phi =  r.flatten(), theta.flatten(), phi.flatten()
        field = np.empty(len(theta))
        for i in range(len(theta)):
            r_idx = np.argmin(np.abs(self.r_lst-r[i])) 
            idx1 = np.searchsorted(self.lat_theta, theta[i])
            if idx1 == 0:
                idx1 = 1
            idx2 = np.searchsorted(self.long_phi, phi[i])
            if idx2 == len(self.long_phi):
                idx2 = idx2 - 1
            field[i] = self.voltage_lst[r_idx][len(self.lat_theta)-idx1, idx2]
        return field
    

    def evaluate_Efield_fast(self,r,theta,phi,J):
        if self.voltage_diff_height_flag is False:
            raise Exception('Calculate the current density first using calc_curr_density() function')
        r, theta, phi =  r.flatten(), theta.flatten(), phi.flatten()
        field = np.empty([len(theta),3])
        curr_density = self.calc_curr_density(r=r[0], J=J)
        for i in range(len(theta)):
            idx1 = np.searchsorted(self.lat_theta, theta[i])
            if idx1 == 0:
                idx1 = 1
            idx2 = np.searchsorted(self.long_phi, phi[i])
            if idx2 == len(self.long_phi):
                idx2 = idx2 - 1
            field[i,0] = curr_density[0][len(self.lat_theta)-idx1, idx2]
            field[i,1] = curr_density[1][len(self.lat_theta)-idx1, idx2]
            field[i,2] = curr_density[2][len(self.lat_theta)-idx1, idx2]
        return field
    
    def calc_voltage_diff_height(self, r_start, J=None, fname_load=None, fname_save=None):
        if fname_load is None:
            dr = 0.0004 #4 um discretization
            self.voltage_lst = []
            self.r_lst = []
            max_iter = 1000
            for i in range(max_iter):
                print("Remaining:", max_iter-i-1)
                voltage_map = self.calc_voltage(r=r_start-i*dr, J=J)
                self.voltage_lst.append(voltage_map)
                self.r_lst.append(r_start-i*dr)
            self.voltage_lst = np.array(self.voltage_lst)
            self.r_lst = np.array(self.r_lst)
            if fname_save is not None:
                np.save(fname_save+"_voltage.npy", self.voltage_lst)
                np.save(fname_save+"_r.npy", self.r_lst)
        else:
            self.voltage_lst=np.load(fname_load+"_voltage.npy")
            self.r_lst=np.load(fname_load+"_r.npy")
        
        self.voltage_diff_height_flag = True
 
    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2), patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        J = np.multiply(J,(elec_radius.reshape(J.shape))**2*np.pi)
        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        plt.show()
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)

        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (cm)', fontsize='19')
        plt.ylabel('y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = curr_density.data*10
        return self.voltage_at_target

    def calc_curr_density(self, r, J=None):
        layer_idx = int(np.where(np.abs(self.radius_vec - np.min(self.radius_vec[self.radius_vec - r*10**(-2) > 0])) < self.eps)[0])
        ## Electrode Density
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs
        # grid_electrode.plot(colorbar='right')
        
        tau_V,tau_Jr , _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))
        
        ## Calculating the voltage        
        voltage_coeff = np.zeros(coeff_array.shape)
        voltage_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        voltage_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        voltage_coeff = shp_harm.SHCoeffs.from_array(voltage_coeff)
        voltage = voltage_coeff.expand(grid='DH')
        self.voltage_at_target = voltage.data*10
        
        lat = voltage.lats()
        lat = lat/180*np.pi

        long = voltage.lons()
        long = long/180*np.pi


        #############################################################################################
        #############################################################################################
        ## Calculating Current Density
        ## Current Density r
        curr_density_r_coeff = np.zeros(coeff_array.shape)
        curr_density_r_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_r_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_r_coeff = shp_harm.SHCoeffs.from_array(curr_density_r_coeff)
        curr_density_r = curr_density_r_coeff.expand(grid='DH')
        self.curr_density_r = curr_density_r.data
        
        ### Current Density Theta
        del_theta = lat[1]-lat[0]
        self.curr_density_theta = np.zeros(self.voltage_at_target.shape)
        self.curr_density_theta[0] = (self.voltage_at_target[1]-self.voltage_at_target[0])/(r*10**(-2)*del_theta)
        self.curr_density_theta[self.curr_density_theta.shape[0]-1] = (self.voltage_at_target[self.curr_density_theta.shape[0]-1] - self.voltage_at_target[self.curr_density_theta.shape[0]-2]) / (r * 10 ** (-2) * del_theta)
        for i in range(self.voltage_at_target.shape[0]-2):
            self.curr_density_theta[i+1] = (-self.voltage_at_target[i]+self.voltage_at_target[i+2])/(2*r*10**(-2)*del_theta)
        self.curr_density_theta = -1*self.cond_vec[layer_idx]*self.curr_density_theta/10
        
        ### Current Density Phi
        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        self.curr_density_phi = np.zeros(self.voltage_at_target.shape)
        cos_theta = np.cos(lat)
        cos_theta[0] = 1
        cos_theta[len(cos_theta)-1]=1
        self.curr_density_phi[:,0] = (self.voltage_at_target[:,1] - self.voltage_at_target[:,0])/(r*10**(-2)*del_phi*cos_theta)
        self.curr_density_phi[:,self.curr_density_theta.shape[1]-1] = (self.voltage_at_target[:,self.curr_density_theta.shape[1]-1] -self.voltage_at_target[:,self.curr_density_theta.shape[1]-2]) / (r*10**(-2)*del_phi*cos_theta)
        for i in range(self.voltage_at_target.shape[1] - 2):
            self.curr_density_phi[:, i+1] = (-self.voltage_at_target[:,i] + self.voltage_at_target[:, i+2]) / (2*r*10**(-2) * del_phi*cos_theta)
        self.curr_density_phi = -1*self.cond_vec[layer_idx]*self.curr_density_phi/10
        self.curr_density_calculated = True
        self.curr_density_at_target = np.sqrt(self.curr_density_theta**2+self.curr_density_phi**2+self.curr_density_r**2)
        return [self.curr_density_r,self.curr_density_theta, self.curr_density_phi]

    def evaluate_field(self,r,x,y, J):
        if self.voltage_diff_height_flag is False:
            raise Exception('Calculate the current density first using calc_curr_density() function')
        self.calc_curr_density(r=r,J=J)

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] <= 0
        x_arr, y_arr = r*10**1*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r*10**1*np.cos(spherical_map_flatten[idx, 1])*np.sin(spherical_map_flatten[idx, 0])
        #r_idx = np.argmin(np.abs(self.r_lst-r)) 
        data_flatten = self.curr_density_at_target.flatten()#self.voltage_lst[r_idx].flatten()
        data_flatten = data_flatten[idx]
        field = np.empty(len(x))
        for i in range(len(x)):
            dist = (x_arr-x[i])**2+(y_arr-y[i])**2
            field[i] = data_flatten[int(np.argmin(dist))]
        return field


    
    def plot_voltage(self, r, J=None, x_limit=None, y_limit=None, fname=None, abs=True):
        if J is None:
           if self.voltage_at_target is False:
                self.calc_voltage(r)
        else:
            self.calc_voltage(r,J)
        return self.plot_given_voltage(r=r, curr_density=self.voltage_at_target, x_limit=x_limit, y_limit=y_limit, fname=fname, abs=abs)

    def plot_given_voltage(self, r, curr_density, x_limit=None, y_limit=None, abs=True, fname=None):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]
        max_v = np.max(np.abs(data_flatten))
        data_flatten = data_flatten[idx]

        if x_limit is None:
            x_limit = np.array((np.min(x), np.max(x)))
        if y_limit is None:
            y_limit = np.array((np.min(y), np.max(y)))

        x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
        y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 100)

        spacing_x = (x_limit[1] - x_limit[0]) / 100
        spacing_y = (y_limit[1] - y_limit[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))

        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")


        labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        plt.xticks(x_ticks, labels_x, fontsize='13')
        plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xlabel('x-axis (mm)', fontsize='17')
        plt.ylabel('y-axis (mm)', fontsize='17')
        plt.title('Voltage ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        if fname is not None:
            plt.savefig(fname)
        plt.tight_layout()
        plt.show()
        print("Position of maximum [%s,%s]"%(max_x,max_y))
        return max_v

    def plot_curr_density(self, r, J=None, x_limit=None, y_limit=None, fname=None, abs=True):
        if J is None:
           if self.curr_density_calculated is False:
                self.calc_curr_density(r)
        else:
            self.calc_curr_density(r,J)
        return self.plot_given_curr_density(r=r, curr_density=self.curr_density_at_target, x_limit=x_limit, y_limit=y_limit, fname=fname, abs=abs)

    def get_max_location(self,  r, J=None):
        if J is None:
           if self.curr_density_calculated is False:
                self.calc_curr_density(r)
        else:
            self.calc_curr_density(r,J)
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(
            spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = self.curr_density_at_target.flatten()
        data_flatten = data_flatten[idx]

        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]
        return [max_x, max_y]



    def plot_given_curr_density(self, r, curr_density, x_limit=None, y_limit=None, abs=True, fname=None):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]

        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]


        if x_limit is None:
            x_limit = np.array((np.min(x), np.max(x)))
        if y_limit is None:
            y_limit = np.array((np.min(y), np.max(y)))

        x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 200)
        y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 200)

        spacing_x = (x_limit[1] - x_limit[0]) / 200
        spacing_y = (y_limit[1] - y_limit[0]) / 200
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))

        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])

        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        if abs is True:
            sns.heatmap(np.abs(data_projected), cmap="jet")
        else:
            sns.heatmap(data_projected, cmap="jet")


        labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        plt.xticks(x_ticks, labels_x, fontsize='13')
        plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xlabel('x-axis (mm)', fontsize='17')
        plt.ylabel('y-axis (mm)', fontsize='17')
        plt.title('Current Density ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)

        plt.show()
        return [max_x, max_y]







