# -------------------------
# Shared simulation parameters
# -------------------------
N = 100
dx = 1.0
steps = 50000
dt = 0.01
save_every = 200
spike_value = 1
stopping_threshold = 1e-4
min_steps = 1000

#Define initiation mode and activator type
#Frequently used:
#init_mode = "random_tight"
#init_mode = "activator_spike"
init_mode = "random_uniform_over0"
#init_mode = "activator_on"
activator_type = "juxtacrine"
#activator_type = "paracrine"

# -------------------------
# Default reaction-diffusion parameters
# -------------------------
params = {
    #fixed by nondimensionalization
    "act_half_sat": 1.0,      # activator half-saturation constant
    "inh_half_sat": 1.0,      # inhibitor half-saturation constant
    "act_decay_rate": 1.0,    # activator decay rate
    "basal_prod": 0.0,        # basal leakiness of production (for both activator and inhibitor)
    "act_diffusion": 1.0,      # if the activator is soluble, diffusion coefficient

    #Free parameters
    "inh_diffusion": 10.0,     # inhibitor diffusion coefficient

    "act_prod_rate": 5.0,    # activator production rate
    "inh_prod_rate": 3.0,     # inhibitor production rate
    "inh_decay_rate": 0.5,    # inhibitor decay rate

    #Hill coefficients
    "act_hill_coeff": 3,      # activator Hill coefficient
    "inh_hill_coeff": 3,      # inhibitor Hill coefficient
}
