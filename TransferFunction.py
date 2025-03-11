import numpy as np

class TransferFunction:

    def __init__(self, cond_vec, radius_vec, max_l=300):

        self.cond_vec = cond_vec
        self.radius_vec = radius_vec*10**(-2) ## cm to m
        self.max_l = max_l
        self.eps = 0.0001*10**(-2)

    def create_frequencies(self, max_nx, max_ny):

        x_frequencies, y_frequencies = np.arange(max_nx + 1), np.arange(max_ny + 1)
        frequencies_X, frequencies_Y = np.meshgrid(x_frequencies, y_frequencies)
        self.frequencies = np.transpose(np.vstack((frequencies_X.flatten(), frequencies_Y.flatten())))
        self.frequencies = self.frequencies[1:]

    def calc_gamma(self):

        no_of_layers = len(self.radius_vec)
        gamma = np.zeros((self.max_l,no_of_layers))
        L = np.arange(self.max_l)+1
        for i in range(no_of_layers-1):

            delta = (self.cond_vec[i]/self.cond_vec[i+1])*(1-gamma[:,i]*((L+1)/L))/(1+gamma[:,i])
            gamma[:,i+1] = (self.radius_vec[i]/self.radius_vec[i+1])**(2*L+1)*(1-delta)/((L+1)/L+delta)

        return gamma, L

    def calc_tauL(self, r):

        r = r*10**(-2) ## cm to m
        layer_idx = int(np.where(np.abs(self.radius_vec-np.min(self.radius_vec[self.radius_vec-r>0]))<self.eps)[0])
        rN = self.radius_vec[len(self.radius_vec)-1]
        condN = self.cond_vec[len(self.radius_vec)-1]
        gamma, L = self.calc_gamma()
        term1 = rN/(condN*(L-(L+1)*gamma[:,len(self.radius_vec)-1]))
        term2 = (1+np.sign(gamma[:,layer_idx])*np.exp((np.log(np.abs(gamma[:,layer_idx]))+np.log((self.radius_vec[layer_idx]/r))*(2*L+1))))*(r/rN)**L
        term2_Jr = ((1/r)*(L - (L+1)*np.sign(gamma[:,layer_idx])*np.exp((np.log(np.abs(gamma[:,layer_idx]))+np.log((self.radius_vec[layer_idx]/r))*(2*L+1)))) * (r / rN)**L)
        term2_Jrr = (L*(L+1) + (L)*(L+1)*np.sign(gamma[:,layer_idx])*np.exp((np.log(np.abs(gamma[:,layer_idx]))+np.log((self.radius_vec[layer_idx]/r))*(2*L+1)))) * (r**(L-2) / rN**L)
        term3 = 1
        for i in range(layer_idx+1, len(self.radius_vec)):
            term3 = term3*((1+np.sign(gamma[:,i])*np.exp(np.log(np.abs(gamma[:,i]))+np.log((self.radius_vec[i]/self.radius_vec[i-1]))*(2*L+1)))/(1+gamma[:,i-1]))
        tau_l = term1*term2*term3
        tau_l_Jr = term1*term2_Jr*term3
        tau_l_Jrr = term1*term2_Jrr*term3

        return tau_l, -1*tau_l_Jr*self.cond_vec[layer_idx], -1*tau_l_Jrr*self.cond_vec[layer_idx]
