# -------------------------
# Shared simulation parameters
# -------------------------
Ny = 100
Nx = 100
dx = 1.0
steps = 10000
dt = 0.01
save_every = 200
stopping_threshold = 1e-4
min_steps = 500

spike_value = 2.0
n_points = 0
noise_amplitude = 0.0
nucleation_rate = 0.01

#Define initiation mode and activator type
#init_mode = "activator_spike"
init_mode = "all_off"
#init_mode = "activator_random_spikes"
activator_type = "juxtacrine"

# -------------------------
# Default reaction-diffusion parameters
# -------------------------
params = {
    #fixed by nondimensionalization
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_decay_rate": 1.0,    # activator decay rate
    "basal_prod": 0.0,        # basal leakiness of production (for both activator and inhibitor)
    "act_diffusion": 0.0,      # if the activator is soluble, diffusion coefficient

    #Free parameters
    "inh_diffusion": 20.0,     # inhibitor diffusion coefficient

    "act_prod_rate": 5.0,    # activator production rate
    "inh_prod_rate": 25.0,     # inhibitor production rate
    "inh_decay_rate": 0.5,    # inhibitor decay rate

    #Hill coefficients
    "act_hill_coeff": 10,      # activator Hill coefficient
    "inh_hill_coeff": 4,      # inhibitor Hill coefficient
}
