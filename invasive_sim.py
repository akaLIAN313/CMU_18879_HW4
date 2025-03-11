from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt

# Load required .hoc files
h.load_file("cells/L23_LBC_cNAC187_1/template.hoc")  # Update with the actual path to the .hoc file
h.load_file("cells/L23_LBC_cNAC187_2/template.hoc")       # Update with the actual path to the .hoc file

# Simulation parameters
n_exc = 10  # Number of excitatory (pyramidal) neurons
n_inh = 5   # Number of inhibitory (PV) neurons
exc_syn_weight = 0.05  # Weight of excitatory synapses
inh_syn_weight = 0.1   # Weight of inhibitory synapses
electrode_current = 0.5  # nA
sim_time = 100  # ms
electrode_position = [0, 0, 0]  # Electrode location in 3D space

# Create realistic neuron class
class RealisticNeuron:
    def __init__(self, template_name, pos):
        if not hasattr(h, template_name):
            raise AttributeError(f"Template '{template_name}' not found in loaded hoc files.")
        self.cell = getattr(h, template_name)()  # Instantiate the cell template
        self.pos = pos  # 3D position of the neuron
        self.synapses = []
        self.set_position()

    def set_position(self):
        """Set the 3D position of the neuron."""
        for sec in self.cell.all:  # Loop through all sections in the cell
            sec.pt3dclear()
            sec.pt3dadd(self.pos[0], self.pos[1], self.pos[2], sec.diam)

    def add_synapse(self, target, weight, delay=1):
        """Add a synapse from this neuron to a target neuron."""
        syn = h.ExpSyn(target.cell.soma(0.5))  # Synapse on the target neuron
        syn.tau = 2 if isinstance(self.cell, h.L23_PC) else 5  # Exc: 2ms, Inh: 5ms
        nc = h.NetCon(self.cell.soma(0.5)._ref_v, syn, sec=self.cell.soma)
        nc.weight[0] = weight
        nc.delay = delay
        self.synapses.append((syn, nc))

# Create excitatory (pyramidal) and inhibitory (PV) neurons
np.random.seed(42)
exc_neurons = [RealisticNeuron("L23_PC", np.random.rand(3) * 500) for _ in range(n_exc)]  # L23 pyramidal cells
inh_neurons = [RealisticNeuron("L23_PV", np.random.rand(3) * 500) for _ in range(n_inh)]  # PV interneurons
all_neurons = exc_neurons + inh_neurons

# Connect neurons randomly
for pre in all_neurons:
    targets = np.random.choice(all_neurons, size=3, replace=False)  # Choose 3 random targets
    for target in targets:
        if isinstance(pre.cell, h.L23_PC):  # Excitatory neuron
            pre.add_synapse(target, exc_syn_weight)
        else:  # Inhibitory neuron
            pre.add_synapse(target, inh_syn_weight)

# Add point electrode stimulation
stim = h.IClamp(exc_neurons[0].cell.soma(0.5))  # Stimulate the first excitatory neuron
stim.delay = 10  # ms
stim.dur = 50    # ms
stim.amp = electrode_current  # nA

# Record membrane potentials
t_vec = h.Vector().record(h._ref_t)  # Time vector
v_vecs = [h.Vector().record(cell.cell.soma(0.5)._ref_v) for cell in all_neurons]  # Voltage vectors

# Run simulation
h.finitialize(-65)  # Initialize membrane potential
h.continuerun(sim_time)  # Run the simulation

# Plot results
plt.figure(figsize=(10, 6))
for i, v in enumerate(v_vecs):
    neuron_type = "Exc" if isinstance(all_neurons[i].cell, h.L23_PC) else "Inh"
    plt.plot(t_vec, v, label=f'Neuron {i+1} ({neuron_type})')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Membrane Potentials")
plt.legend()
plt.show()
