import numpy as np
import time

class ElecGrid:

    def __init__(self, max_l, r_max):
        self.eps = 0.0001*10**(-2)
        self.max_l = max_l
        self.r_max = r_max*10**(-2)

        self.N_lat = 2*self.max_l+2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / self.N_long)
        self.lat_theta = np.arange(-np.pi/2.0, np.pi / 2.0, np.pi / self.N_lat)


        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl,long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))


    def elec_pos(self,elec_theta, elec_phi, elec_radii, via_size=None):
        lat_diff = self.electrode_spherical_map_flatten[:,0]-elec_theta
        long_diff = self.electrode_spherical_map_flatten[:,1]-elec_phi

        dist = np.sin(lat_diff/2.0)**2+np.cos(self.electrode_spherical_map_flatten[:,0])*np.cos(elec_theta*np.ones(len(self.electrode_spherical_map_flatten[:,0])))*np.sin(long_diff/2.0)**2
        dist = self.r_max*2*np.arcsin(np.sqrt(dist))
        if via_size is None:
            idx = dist<elec_radii
            elec_thetaPhi = self.electrode_spherical_map_flatten[idx]
        else:
            idx = dist < elec_radii
            elec_thetaPhi = self.electrode_spherical_map_flatten[idx]
            dist = dist[idx]

            idx = dist>via_size
            elec_thetaPhi = elec_thetaPhi[idx]

        return elec_thetaPhi

    def conc_ring(self, center, radii, width):

        lat_diff = self.electrode_spherical_map_flatten[:, 0] - center[0]
        long_diff = self.electrode_spherical_map_flatten[:, 1] - center[1]
        dist = np.sin(lat_diff / 2.0) ** 2 + np.cos(self.electrode_spherical_map_flatten[:, 0]) * np.cos(center[0] * np.ones(len(self.electrode_spherical_map_flatten[:, 0]))) * np.sin(long_diff / 2.0) ** 2
        dist = self.r_max * 2 * np.arcsin(np.sqrt(dist))
        idx_1 = np.array(dist <= radii)
        idx_2 = np.array(dist >= radii-width)
        idx = idx_1*idx_2
        elec_thetaPhi = self.electrode_spherical_map_flatten[idx]

        return elec_thetaPhi

    def electrode_sampling(self, elec_pos, elec_radii, injected_curr, via_size=None): ### Driscol and Healy Sampling
        ### Conversion of elec_radius to spherical coordinate.
        ### 2*pi*r_max**2*(1-cos(e_sphere/r_max)) = pi*e_act**2
        ### rmax*arccos(1-e_act**2/(2*r_act**2))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        elec_radii = elec_radii*10**(-2) ## cm to m
        if via_size is not None:
            via_size = via_size*10**(-2)
        elec_radii = self.r_max*np.arccos(1-(elec_radii**2/(2.0*self.r_max**2)))
        start_time = time.time()
        self.elec_thetaPhi = []
        for i in range(len(elec_pos)):
            if via_size is None:
                self.elec_thetaPhi.append(self.elec_pos(elec_pos[i,0],elec_pos[i,1], elec_radii[i]))
            else:
                self.elec_thetaPhi.append(self.elec_pos(elec_pos[i, 0], elec_pos[i, 1], elec_radii[i], via_size=via_size[i]))

        #print("Time Taken for finding electrode Position:", time.time()-start_time)
        start_time = time.time()
        for i in range(len(self.elec_thetaPhi)):
            for ii in range(len(self.elec_thetaPhi[i])):
                idx1 = np.searchsorted(self.lat_theta, self.elec_thetaPhi[i][ii,0])
                if idx1==len(self.lat_theta):
                    idx1=idx1-1
                idx2 = np.searchsorted(self.long_phi,self.elec_thetaPhi[i][ii,1])
                if idx2==len(self.long_phi):
                    idx2=idx2-1

                self.electrode_spherical_map[idx1,idx2]=injected_curr[i]


    def sample_conc_ring(self, center=None, radius=None, width=None, injected_curr=None):
        ### Conversion of elec_radius to spherical coordinate.
        ### 2*pi*r_max**2*(1-cos(e_sphere/r_max)) = pi*e_act**2
        ### rmax*arccos(1-e_act**2/(2*r_act**2))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        radius = radius*10**(-2)  ## cm to m
        width = width*10**(-2)
        start_time = time.time()
        self.elec_thetaPhi = []
        for i in range(len(radius)):
            self.elec_thetaPhi.append(self.conc_ring(center=center[i], radii=radius[i], width=width[i]))

        # print("Time Taken for finding electrode Position:", time.time()-start_time)
        start_time = time.time()
        for i in range(len(self.elec_thetaPhi)):
            for ii in range(len(self.elec_thetaPhi[i])):
                idx1 = np.searchsorted(self.lat_theta, self.elec_thetaPhi[i][ii, 0])
                if idx1 == len(self.lat_theta):
                    idx1 = idx1 - 1
                idx2 = np.searchsorted(self.long_phi, self.elec_thetaPhi[i][ii, 1])
                if idx2 == len(self.long_phi):
                    idx2 = idx2 - 1

                self.electrode_spherical_map[idx1, idx2] = injected_curr[i]


    def __call__(self, elec_pos, elec_radii, injected_curr, *args, **kwargs):

        self.electrode_sampling(elec_pos, elec_radii, injected_curr)
        return np.flip(self.electrode_spherical_map, axis=0)


class PreDefElecGrid(ElecGrid):

    ## Creates a ring like pattern
    def uniform_sampling_north_pole(self, elec_spacing, elec_radius, patch_size):
        elec_spacing = elec_spacing*10**(-2)
        patch_size = patch_size*10**(-2)
        elec_radius= elec_radius*10**(-2)
        if elec_spacing < 2*elec_radius:
            raise Exception('Electrodes Overlap')
        theta_loc = np.arange(np.pi/2.0, np.pi/2.0-patch_size/self.r_max, -1*elec_spacing/self.r_max)
        theta_loc = theta_loc[1:]
        self.elec_loc = np.array((np.pi/2.0, 0)).reshape(1, 2)
        for i in range(len(theta_loc)):
            r_ring = self.r_max*np.cos(theta_loc[i])
            num_elec = np.floor(2*np.pi*r_ring/elec_spacing)
            phi_loc = np.arange(0, np.pi * 2, 2*np.pi/num_elec)
            temp = np.transpose(np.array((theta_loc[i] * np.ones(len(phi_loc)), phi_loc)))
            self.elec_loc = np.vstack((self.elec_loc, temp))
        self.ground_elec = np.array((-np.pi/2.0,0),)

        return self.elec_loc, self.ground_elec, np.ones(len(self.elec_loc))*elec_radius*10**(2)

    def uniform_sampling_north_pole_extra(self, elec_spacing, elec_radius, patch_size, neg_elec_pos, neg_elec_radius):
        elec_spacing = elec_spacing*10**(-2)
        patch_size = patch_size*10**(-2)
        elec_radius= elec_radius*10**(-2)
        if elec_spacing < 2*elec_radius:
            raise Exception('Electrodes Overlap')
        theta_loc = np.arange(np.pi/2.0, np.pi/2.0-patch_size/self.r_max, -1*elec_spacing/self.r_max)
        theta_loc = theta_loc[1:]
        self.elec_loc = np.array((np.pi/2.0, 0)).reshape(1, 2)
        for i in range(len(theta_loc)):
            r_ring = self.r_max*np.cos(theta_loc[i])
            num_elec = np.floor(2*np.pi*r_ring/elec_spacing)
            phi_loc = np.arange(0, np.pi * 2, 2*np.pi/num_elec)
            temp = np.transpose(np.array((theta_loc[i] * np.ones(len(phi_loc)), phi_loc)))
            self.elec_loc = np.vstack((self.elec_loc, temp))

        self.elec_loc = np.vstack((self.elec_loc, neg_elec_pos.reshape(1,2)))
        self.ground_elec = np.array((0,np.pi/4),)
        elec_radius_lst = np.hstack((np.ones(len(self.elec_loc)-1)*elec_radius*10**(2),[neg_elec_radius]))

        return self.elec_loc, self.ground_elec, elec_radius_lst

    def approx_rectangular(self, elec_spacing, elec_radius, num_elec):

        elec_spacing = elec_spacing * 10 ** (-2)
        elec_radius = elec_radius * 10 ** (-2)
        if elec_spacing < 2*elec_radius:
            raise Exception('Electrodes Overlap')

        ### Cartesian Plane
        num_elec_x, num_elec_y = num_elec[0], num_elec[1]
        half_num_x = (num_elec_x-1)/2
        x_pos = np.arange(-half_num_x*elec_spacing, (half_num_x+1)*elec_spacing, elec_spacing)

        half_num_y = (num_elec_y - 1) / 2
        y_pos = np.arange(-half_num_y * elec_spacing, (half_num_y + 1) * elec_spacing, elec_spacing)

        X, Y = np.meshgrid(x_pos, y_pos)
        X, Y = X.flatten(), Y.flatten()


        ### Projection to the sphere
        Z = np.sqrt(self.r_max**2-X**2-Y**2)

        theta_loc = np.arccos(Z/self.r_max)
        theta_loc = np.pi/2-theta_loc

        phi_loc = np.zeros(len(X))
        for i in range(len(X)):
            if np.abs(X[i])<1e-7:
                if Y[i]>=0:
                    phi_loc[i] =np.pi/2
                else:
                    phi_loc[i] = 3*np.pi/2
            elif np.abs(Y[i])<1e-7:
                if X[i]>=0:
                    phi_loc[i] =0
                else:
                    phi_loc[i] = np.pi
            elif X[i]>0 and Y[i]>0:
                phi_loc[i] = np.arctan(Y[i]/X[i])
            elif X[i]<0 and Y[i]>0:
                phi_loc[i] = np.pi-np.arctan(Y[i]/(-1*X[i]))
            elif X[i]<0 and Y[i]<0:
                phi_loc[i] = np.pi+np.arctan(Y[i]/X[i])
            elif X[i]>0 and Y[i]<0:
                phi_loc[i] = 2*np.pi-np.arctan((-1*Y[i])/X[i])
        self.elec_loc = np.transpose(np.vstack((theta_loc, phi_loc)))
        self.ground_elec = np.array((-np.pi/2.0,0),)

        return self.elec_loc, self.ground_elec, np.ones(len(self.elec_loc))*elec_radius*10**(2)

    def NHP_patch(self, elec_radius, via_size, elec_spacing, num_elec, elec_patch_angle):

        elec_spacing = elec_spacing * 10 ** (-2)
        elec_radius = elec_radius * 10 ** (-2)
        via_size = via_size*10**(-2)

        if elec_spacing < 2 * elec_radius:
            raise Exception('Electrodes Overlap')

        num_elec_y = len(num_elec)
        elec_loc_XY = []

        half_num_y = (num_elec_y-1)/2.0
        pitch = np.arange(-half_num_y * elec_spacing, (half_num_y + 1) * elec_spacing, elec_spacing)
        for i in range(len(pitch)):
            num_elec_x = num_elec[i]
            half_num_x = (num_elec_x - 1) / 2
            start_x = np.arange(-half_num_x * elec_spacing, (half_num_x + 1) * elec_spacing, elec_spacing)
            x_temp = start_x + pitch[i]*np.cos(elec_patch_angle)
            y_temp = pitch[i]*np.sin(elec_patch_angle)*np.ones(len(start_x))
            elec_loc_temp = np.hstack((x_temp.reshape(-1,1), y_temp.reshape(-1,1)))
            if i == 0:
                elec_loc_XY = elec_loc_temp
            else:
                elec_loc_XY = np.vstack((elec_loc_XY, elec_loc_temp))

        X, Y = elec_loc_XY[:,0], elec_loc_XY[:,1]
        ### Projection to the sphere
        Z = np.sqrt(self.r_max ** 2 - X ** 2 - Y ** 2)

        theta_loc = np.arccos(Z / self.r_max)
        theta_loc = np.pi / 2 - theta_loc

        phi_loc = np.zeros(len(X))
        for i in range(len(X)):
            if np.abs(X[i]) < 1e-7:
                if Y[i] >= 0:
                    phi_loc[i] = np.pi/2
                else:
                    phi_loc[i] = 3*np.pi/2
            elif np.abs(Y[i]) < 1e-7:
                if X[i] >= 0:
                    phi_loc[i] = 0
                else:
                    phi_loc[i] = np.pi
            elif X[i] > 0 and Y[i] > 0:
                phi_loc[i] = np.arctan(Y[i] / X[i])
            elif X[i] < 0 and Y[i] > 0:
                phi_loc[i] = np.pi - np.arctan(Y[i] / (-1 * X[i]))
            elif X[i] < 0 and Y[i] < 0:
                phi_loc[i] = np.pi + np.arctan(Y[i] / X[i])
            elif X[i] > 0 and Y[i] < 0:
                phi_loc[i] = 2 * np.pi - np.arctan((-1 * Y[i]) / X[i])
        self.elec_loc = np.transpose(np.vstack((theta_loc, phi_loc)))

        self.ground_elec = np.array((-np.pi / 2.0, 0), )

        return self.elec_loc, self.ground_elec, np.ones(len(self.elec_loc)) * elec_radius * 10 ** (2), np.ones(len(self.elec_loc)) * via_size * 10 ** (2)

    def NHP_patch_big_elec(self, elec_radius, via_size, elec_spacing, num_elec, elec_patch_angle, big_elec_pos=None, big_elec_radius=None):

        elec_spacing = elec_spacing * 10 ** (-2)
        elec_radius = elec_radius * 10 ** (-2)
        via_size = via_size*10**(-2)

        if elec_spacing < 2 * elec_radius:
            raise Exception('Electrodes Overlap')

        num_elec_y = len(num_elec)
        elec_loc_XY = []

        half_num_y = (num_elec_y-1)/2.0
        pitch = np.arange(-half_num_y * elec_spacing, (half_num_y + 1) * elec_spacing, elec_spacing)
        for i in range(len(pitch)):
            num_elec_x = num_elec[i]
            half_num_x = (num_elec_x - 1) / 2
            start_x = np.arange(-half_num_x * elec_spacing, (half_num_x + 1) * elec_spacing, elec_spacing)
            x_temp = start_x + pitch[i]*np.cos(elec_patch_angle)
            y_temp = pitch[i]*np.sin(elec_patch_angle)*np.ones(len(start_x))
            elec_loc_temp = np.hstack((x_temp.reshape(-1,1), y_temp.reshape(-1,1)))
            if i == 0:
                elec_loc_XY = elec_loc_temp
            else:
                elec_loc_XY = np.vstack((elec_loc_XY, elec_loc_temp))

        X, Y = elec_loc_XY[:,0], elec_loc_XY[:,1]

        ### Projection to the sphere
        Z = np.sqrt(self.r_max ** 2 - X ** 2 - Y ** 2)

        theta_loc = np.arccos(Z / self.r_max)
        theta_loc = np.pi / 2 - theta_loc

        phi_loc = np.zeros(len(X))
        for i in range(len(X)):
            if np.abs(X[i]) < 1e-7:
                if Y[i] >= 0:
                    phi_loc[i] = 0
                else:
                    phi_loc[i] = np.pi
            elif np.abs(Y[i]) < 1e-7:
                if X[i] >= 0:
                    phi_loc[i] = np.pi / 2
                else:
                    phi_loc[i] = 3 * np.pi / 2
            elif X[i] > 0 and Y[i] > 0:
                phi_loc[i] = np.arctan(Y[i] / X[i])
            elif X[i] < 0 and Y[i] > 0:
                phi_loc[i] = np.pi - np.arctan(Y[i] / (-1 * X[i]))
            elif X[i] < 0 and Y[i] < 0:
                phi_loc[i] = np.pi + np.arctan(Y[i] / X[i])
            elif X[i] > 0 and Y[i] < 0:
                phi_loc[i] = 2 * np.pi - np.arctan((-1 * Y[i]) / X[i])
        self.elec_loc = np.transpose(np.vstack((theta_loc, phi_loc)))
        self.elec_loc = np.vstack((self.elec_loc, big_elec_pos.reshape(-1, 2)))
        self.ground_elec = np.array((-np.pi / 2.0, 0), )
        elec_radius_lst = np.hstack((np.ones(len(self.elec_loc) - len(big_elec_radius)) * elec_radius * 10 ** (2), big_elec_radius))
        via_size_lst = np.hstack((np.ones(len(self.elec_loc)-len(big_elec_radius)) * via_size * 10 ** (2), np.zeros(len(big_elec_radius))))
        return self.elec_loc, self.ground_elec, elec_radius_lst, via_size_lst

    def rodent_patch(self, elec_radius, via_size, elec_spacing, num_elec):
        
        elec_spacing = elec_spacing * 10 ** (-2)
        elec_radius = elec_radius * 10 ** (-2)
        via_size = via_size*10**(-2)

        if elec_spacing < 2 * elec_radius:
            raise Exception('Electrodes Overlap')

        bregma_x, bregma_y = 0, -2.89*10**(-3)
        bregma_idx = 1

        #num_elec_y = len(num_elec)
        #elec_loc_XY = []

        y_loc = np.empty(len(num_elec))
        y_loc[1] = bregma_y
        for i in range(y_loc.shape[0]):
            y_loc[i] = y_loc[bregma_idx]-(bregma_idx-i)*self.r_max*np.sin(np.sqrt(3)*elec_spacing/(2.0*self.r_max))
        for i in range(y_loc.shape[0]):
            num_elec_x = num_elec[i]
            half_num_x = (num_elec_x - 1) / 2
            start_x = np.arange(-half_num_x * elec_spacing, (half_num_x + 1) * elec_spacing, elec_spacing)
            x_temp = np.empty(start_x.shape)
            for j in range(start_x.shape[0]):
                x_temp[j] = self.r_max*np.sin(start_x[j]/self.r_max)+bregma_x
            y_temp = y_loc[i]*np.ones(len(start_x))
            elec_loc_temp = np.hstack((x_temp.reshape(-1,1), y_temp.reshape(-1,1)))
            if i == 0:
                elec_loc_XY = elec_loc_temp
            else:
                elec_loc_XY = np.vstack((elec_loc_XY, elec_loc_temp))
        X, Y = elec_loc_XY[:,0], elec_loc_XY[:,1]
        ### Projection to the sphere
        Z = np.sqrt(self.r_max ** 2 - X ** 2 - Y ** 2)
        theta_loc = np.arccos(Z / self.r_max)
        theta_loc = np.pi / 2 - theta_loc
        phi_loc = np.zeros(len(X))
        
        for i in range(len(X)):
            if np.abs(X[i]) < 1e-7:
                if Y[i] >= 0:
                    phi_loc[i] = np.pi/2
                else:
                    phi_loc[i] = 3*np.pi/2
            elif np.abs(Y[i]) < 1e-7:
                if X[i] >= 0:
                    phi_loc[i] = 0
                else:
                    phi_loc[i] = np.pi
            elif X[i] > 0 and Y[i] > 0:
                phi_loc[i] = np.arctan(Y[i] / X[i])
            elif X[i] < 0 and Y[i] > 0:
                phi_loc[i] = np.pi - np.arctan(Y[i] / (-1 * X[i]))
            elif X[i] < 0 and Y[i] < 0:
                phi_loc[i] = np.pi + np.arctan(Y[i] / X[i])
            elif X[i] > 0 and Y[i] < 0:
                phi_loc[i] = 2 * np.pi - np.arctan((-1 * Y[i]) / X[i])
        self.elec_loc = np.transpose(np.vstack((theta_loc, phi_loc)))
        self.ground_elec = np.array((-np.pi / 2.0, 0), )

        return self.elec_loc, self.ground_elec, np.ones(len(self.elec_loc)) * elec_radius * 10 ** (2), np.ones(len(self.elec_loc)) * via_size * 10 ** (2)

    def conc_ring_patch(self, spacing=None, width=None, num_rings=None, start_radius=None, center=None):

        spacing = spacing*10**(-2)
        width = width*10**(-2)
        start_radius = start_radius*10**(-2)
        radii = np.arange(start_radius, num_rings * spacing + start_radius, spacing)
        self.ground_elec = np.array((-np.pi / 2.0, 0),)
        return radii, width*np.ones(len(radii)), center, self.ground_elec



    def custom_grid(self, elec_radius, theta_elec, phi_elec):
        theta_elec = np.array(theta_elec).flatten()
        phi_elec = np.array(phi_elec).flatten()
        elec_radius =elec_radius*10**(-2)
        self.ground_elec = np.array((-np.pi/2.0,0),)
        self.elec_pos = np.transpose(np.vstack((theta_elec,phi_elec)))
        return self.elec_loc, self.ground_elec, np.ones(len(self.elec_loc)) * elec_radius * 10 ** (2)

    ## Edit Todo : Creates approximately uniform collection of points using the Fibonacci Sphere Algorithm
