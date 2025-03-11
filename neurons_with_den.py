import math
import random
import matplotlib.pyplot as plt
from neuron import h, gui
import numpy as np

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
num_exc = 400
num_inh = 100
total_neurons = num_exc + num_inh

radius = 50  # radius of the spherical surface in micrometers
random.seed(42)  # for reproducibility

# Basic HH parameters
Ra = 150       # Axial resistance (Ohm*cm)
Rm = 20_000    # Membrane resistance (Ohm*cm^2)
Cm = 1         # Membrane capacitance (µF/cm^2)
v_rest = -70   # Resting membrane potential (mV)

# Conductances for excitatory neurons
exc_gnabar = 0.12
exc_gkbar = 0.036
exc_gl = 0.0003
exc_el = -54.3

# Conductances for inhibitory neurons
inh_gnabar = 0.10
inh_gkbar = 0.04
inh_gl = 0.0004
inh_el = -60

# ------------------------------------------------------------
# Neuron Creation with Dendrites
# ------------------------------------------------------------
def create_cell(is_exc=True):
    # Create soma
    soma = h.Section(name='soma')
    soma.L = 20     # Length in micrometers
    soma.diam = 20  # Diameter in micrometers
    soma.cm = Cm
    soma.Ra = Ra
    soma.insert('pas')
    soma.g_pas = 1.0 / Rm
    soma.e_pas = v_rest

    # Insert Hodgkin-Huxley mechanism
    soma.insert('hh')
    if is_exc:
        soma.gnabar_hh = exc_gnabar
        soma.gkbar_hh = exc_gkbar
        soma.gl_hh = exc_gl
        soma.el_hh = exc_el
    else:
        soma.gnabar_hh = inh_gnabar
        soma.gkbar_hh = inh_gkbar
        soma.gl_hh = inh_gl
        soma.el_hh = inh_el

    # Create dendrite
    dend = h.Section(name='dend')
    dend.L = 100    # Length in micrometers
    dend.diam = 2   # Diameter in micrometers
    dend.cm = Cm
    dend.Ra = Ra
    dend.insert('pas')
    dend.g_pas = 1.0 / Rm
    dend.e_pas = v_rest

    # Connect dendrite to soma
    dend.connect(soma, 1.0, 0.0)  # Connect end of soma to start of dendrite

    # Insert HH mechanism in dendrite (optional, can be simplified)
    dend.insert('hh')
    if is_exc:
        dend.gnabar_hh = exc_gnabar * 0.1  # Reduced conductances in dendrites
        dend.gkbar_hh = exc_gkbar * 0.1
        dend.gl_hh = exc_gl * 1.5
        dend.el_hh = exc_el
    else:
        dend.gnabar_hh = inh_gnabar * 0.1
        dend.gkbar_hh = inh_gkbar * 0.1
        dend.gl_hh = inh_gl * 1.5
        dend.el_hh = inh_el

    return soma, dend

# Create a shuffled list of cell types
cell_types = ['exc'] * num_exc + ['inh'] * num_inh
random.shuffle(cell_types)

cells = []
for ctype in cell_types:
    if ctype == 'exc':
        cells.append(create_cell(is_exc=True))
    else:
        cells.append(create_cell(is_exc=False))

# Generate random positions in a small spherical cap (0 ≤ θ ≤ π/4)
positions = []
for i in range(total_neurons):
    phi = 2 * math.pi * random.random()
    u = random.uniform(math.cos(math.pi / 4), 1.0)  # cos(theta) in [cos(π/4),1]
    theta = math.acos(u)
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    positions.append((x, y, z))

# Define the stimulated neuron (e.g., neuron 0)
stim_position = positions[0]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 +
                     (p1[1] - p2[1]) ** 2 +
                     (p1[2] - p2[2]) ** 2)

# Stimulation Parameters
I_max = 0.3       # Maximum amplitude of injected current (nA)
sigma = 50.0      # Spatial decay parameter (µm)
stim_delay = 100  # Delay before stimulation starts (ms)
stim_dur = 200    # Duration of stimulation (ms)

stim_list = []
for i, cell in enumerate(cells):
    d = distance(positions[i], stim_position)
    I_amp = I_max * math.exp(-(d ** 2) / (sigma ** 2))

    stim = h.IClamp(cell[0](0.5))  # Inject current at the soma midpoint
    stim.delay = stim_delay
    stim.dur = stim_dur
    stim.amp = I_amp
    stim_list.append(stim)

# Identify excitatory and inhibitory cells
exc_indices = [i for i, ctype in enumerate(cell_types) if ctype == 'exc']
inh_indices = [i for i, ctype in enumerate(cell_types) if ctype == 'inh']

exc_cells = [cells[i] for i in exc_indices]
inh_cells = [cells[i] for i in inh_indices]

# Add synapses (Exc → Inh)
synapses = []
for i_inh in inh_indices:
    inh_cell = cells[i_inh][0]  # Access soma for synapse placement
    syn = h.ExpSyn(inh_cell(0.5))
    syn.tau = 2.0
    syn.e = 0.0  # Excitatory reversal potential
    synapses.append((i_inh, syn))

netcons = []
weight = 0.002
delay = 1.0
threshold = 0.0
for i_exc in exc_indices:
    exc_cell = cells[i_exc][0]  # Access soma for spike detection
    for (i_inh, syn) in synapses:
        nc = h.NetCon(exc_cell(0.5)._ref_v, syn, sec=exc_cell)
        nc.threshold = threshold
        nc.weight[0] = weight
        nc.delay = delay
        netcons.append(nc)

# Record from all cells
t_vec = h.Vector().record(h._ref_t)
v_vecs = []
for i, cell in enumerate(cells):
    v_vec = h.Vector()
    v_vec.record(cell[0](0.5)._ref_v)  # Record soma voltage
    v_vecs.append((i, v_vec))

# Optionally, record dendritic voltages
# Uncomment the following lines to record dendritic voltages
# dend_v_vecs = []
# for i, cell in enumerate(cells):
#     dend_v = h.Vector()
#     dend_v.record(cell[1](0.5)._ref_v)  # Record dendrite midpoint voltage
#     dend_v_vecs.append((i, dend_v))

# Run simulation
h.finitialize(v_rest)
h.continuerun(400)
t = np.array(t_vec.to_python())

# Spike Detection and Firing Rate Computation
def detect_spikes(voltage_trace, time, threshold=0.0):
    spikes = []
    above = voltage_trace > threshold
    for i in range(1, len(voltage_trace)):
        if above[i] and not above[i - 1]:
            spikes.append(time[i])
    return np.array(spikes)

firing_rates = []
sim_duration = t[-1] - t[0]  # total simulation time in ms
sim_time_sec = sim_duration / 1000.0  # convert to seconds

all_voltages = []
for i, (idx, v_vector) in enumerate(v_vecs):
    v = np.array(v_vector.to_python())
    all_voltages.append(v)
    spikes = detect_spikes(v, t, threshold=0.0)
    spike_count = len(spikes)
    firing_rate = spike_count / sim_time_sec  # spikes per second (Hz)
    firing_rates.append(firing_rate)

firing_rates = np.array(firing_rates)

# ------------------------------------------------------------
# Plot voltage traces in a raster-like format
# We'll offset each neuron's trace so they don't overlap
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))
offset = 100  # vertical offset between neurons
for i, v in enumerate(all_voltages):
    # Shift each neuron's trace by i*offset
    shifted_v = v + i * offset
    color = 'blue' if cell_types[i] == 'exc' else 'red'
    plt.plot(t, shifted_v, color=color, linewidth=0.5)

plt.axvspan(stim_delay, stim_delay + stim_dur, color='gray', alpha=0.3, label='Stimulation Period')
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Neuron (offset by 100 mV each)', fontsize=14)
plt.title('Voltage Traces of 500 Neurons with Dendrites', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot firing rate on neuron positions (3D)
# ------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x_coords = [p[0] for p in positions]
y_coords = [p[1] for p in positions]
z_coords = [p[2] for p in positions]

scatter = ax.scatter(x_coords, y_coords, z_coords, c=firing_rates, cmap='viridis', s=20)
ax.scatter(positions[0][0], positions[0][1], positions[0][2],
           c='green', s=100, marker='X', label='Stimulated Neuron 0')
ax.set_xlabel('X (µm)', fontsize=12)
ax.set_ylabel('Y (µm)', fontsize=12)
ax.set_zlabel('Z (µm)', fontsize=12)
ax.set_title('Neuron Locations and Firing Rates (500 Neurons)', fontsize=14)
ax.legend()
cb = fig.colorbar(scatter, ax=ax, shrink=0.6)
cb.set_label('Firing Rate (Hz)', fontsize=12)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot neuron locations by type (3D)
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue' if ctype == 'exc' else 'red' for ctype in cell_types]
ax.scatter(x_coords, y_coords, z_coords, c=colors, s=20, alpha=0.6)
ax.scatter(positions[0][0], positions[0][1], positions[0][2],
           c='green', s=100, marker='X', label='Stimulated Neuron 0')

ax.set_xlabel('X (µm)', fontsize=12)
ax.set_ylabel('Y (µm)', fontsize=12)
ax.set_zlabel('Z (µm)', fontsize=12)
ax.set_title('Neuron Locations by Type (500 Neurons)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
