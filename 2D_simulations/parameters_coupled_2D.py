# -------------------------
# Shared simulation parameters
# -------------------------
Ny = 100
Nx = 100
dx = 1.0
steps = 5000
dt = 0.01
save_every = 200
stopping_threshold = 1e-4
min_steps = 500

spike_value = 2.0
nucleation_rate = 0.01
#PARAMETERS USED EXCLUSIVELY FOR simuations with induced CO-INITIATION
n_points = 0 #number of points for initial co-activation
set_peak_height = 20 #acvtivator level at these initial points

#Define initiation mode and activator type
#init_mode = "activator_spike"
init_mode = "activator_random_spikes"
#init_mode = "activator_random_spikes"
activator_type = "juxtacrine"

# -------------------------
# Default reaction-diffusion parameters
# -------------------------
p1 = {
    #fixed by nondimensionalization
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_decay_rate": 1.0,    # activator decay rate
    "basal_prod": 0.0,        # basal leakiness of production (for both activator and inhibitor)
    "act_diffusion": 1.0,      # if the activator is soluble, diffusion coefficient

    #Free parameters
    "inh_diffusion": 20.0,     # inhibitor diffusion coefficient

    "act_prod_rate": 5.0,    # activator production rate
    "inh_prod_rate": 8.0,     # inhibitor production rate
    "inh_decay_rate": 0.5,    # inhibitor decay rate
    #Cross inhibition
    "cross_inhibition_rate": 0.0,

    #Hill coefficients
    "act_hill_coeff": 4,      # activator Hill coefficient
    "inh_hill_coeff": 4,      # inhibitor Hill coefficient
}

p2 = {
    #fixed by nondimensionalization
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_decay_rate": 1.0,    # activator decay rate
    "basal_prod": 0.0,        # basal leakiness of production (for both activator and inhibitor)
    "act_diffusion": 1.0,      # if the activator is soluble, diffusion coefficient

    #Free parameters
    "inh_diffusion": 20.0,     # inhibitor diffusion coefficient

    "act_prod_rate": 6.0,    # activator production rate
    "inh_prod_rate": 25.0,     # inhibitor production rate
    "inh_decay_rate": 0.5,    # inhibitor decay rate
    #Cross inhibition
    "cross_inhibition_rate": 0.0,

    #Hill coefficients
    "act_hill_coeff": 10,      # activator Hill coefficient
    "inh_hill_coeff": 4,      # inhibitor Hill coefficient
}
