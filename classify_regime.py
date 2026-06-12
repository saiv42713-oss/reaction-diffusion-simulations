import numpy as np
from simulation_2D import run_coupled_hex
from parameters_2D import params as ben_params
from radial_autocor_npz import autocorrelation_2d, analyze_image_autocorrelation
# Base simulation settings matching Fig 1C from the paper
SIM_PARAMS = {
    "Ny": 100,           # grid height (smaller for speed)
    "Nx": 100,           # grid width
    "steps": 50000,     # max steps
    "dt": 0.01,
    "dx": 1.0,
    "stopping_threshold": 1e-6,  # much stricter — don't stop early
    "min_steps": 5000,            # force at least 5000 steps
    "save_every": 1000,
    "spike_value": 2.0,
}

# Fixed biological parameters matching Fig 1C
BASE_PARAMS = {
    "act_half_sat": 1.0,    # fixed — was 3.0, wrong
    "inh_half_sat": 1.0,    # fixed — was 1.0, wrong
    "act_decay_rate": 1.0,
    "inh_decay_rate": 0.5,    # was 1.0 — this was killing the steady state
    "act_diffusion": 0.0,
    "inh_diffusion": 10.0,
    "basal_prod": 0.0,
    "act_hill_coeff": 3,
    "inh_hill_coeff": 3,
}

np.random.seed(42)
def classify_regime(act_prod_rate, inh_prod_rate):
    params = {**BASE_PARAMS,
              "act_prod_rate": act_prod_rate,
              "inh_prod_rate": inh_prod_rate}
   
    # Run the first simulation
    A_hist, R_hist, final_step, a_ss, i_ss = run_coupled_hex(
    SIM_PARAMS["Ny"], SIM_PARAMS["Nx"],
    SIM_PARAMS["steps"], SIM_PARAMS["dt"], SIM_PARAMS["dx"],
    params,
    SIM_PARAMS["stopping_threshold"],
    SIM_PARAMS["min_steps"],
    init_mode="random_uniform_over0",
    activator_type="juxtacrine",
    spike_value=SIM_PARAMS["spike_value"],
    save_every=SIM_PARAMS["save_every"]
    )
    final_A = A_hist[-1]
   
    # Modify params for the second simulation
    A_hist, R_hist, final_step, a_ss, i_ss = run_coupled_hex(
    SIM_PARAMS["Ny"], SIM_PARAMS["Nx"],
    SIM_PARAMS["steps"], SIM_PARAMS["dt"], SIM_PARAMS["dx"],
    params,
    SIM_PARAMS["stopping_threshold"],
    SIM_PARAMS["min_steps"],
    init_mode="spike_steady_state",
    activator_type="juxtacrine",
    spike_value=SIM_PARAMS["spike_value"],
    save_every=SIM_PARAMS["save_every"]
    )
    final_A_spike = A_hist[-1]
    # Check for ON state
    if np.mean(final_A) > 0.1 and np.std(final_A) < 0.01:
        return "ON"
    # DEBUG
    print(f"final_A_spike mean: {np.mean(final_A_spike):.4f}")
    print(f"final_A_spike max: {np.max(final_A_spike):.4f}")
    print(f"a_ss: {a_ss:.4f}")
    print(f"final_A mean: {np.mean(final_A):.4f}")
    print(f"final_A max: {np.max(final_A):.4f}")
    fft_debug = np.abs(np.fft.fft2(final_A))
    fft_debug[0, 0] = 0
    peak_debug = np.max(fft_debug)
    mean_debug = np.mean(fft_debug)
    print(f"FFT peak: {peak_debug:.2f}, FFT mean: {mean_debug:.2f}, ratio: {peak_debug/mean_debug:.2f}")

    #Check 1 — TURING: Run analyze_image_autocorrelation(final_A) — it returns a DataFrame with a radial_autocorrelation column. If that curve goes below zero at any point → oscillation → return "TURING"

    if np.mean(final_A) > 0.1 and np.std(final_A) > 0.1:
        A_2d, df = analyze_image_autocorrelation(final_A)
        radial_ac = df["radial_autocorrelation"].values
        if np.any(radial_ac < 0):
            # Secondary check — sharp FFT peak confirms Turing vs Irregular
            fft = np.abs(np.fft.fft2(final_A))
            fft[0, 0] = 0
            peak = np.max(fft)
            mean_fft = np.mean(fft)
            if peak / mean_fft > 40:
                return "TURING"
            else:
                return "IRREGULAR"
    #Check 2 — OFF: mean of final_A_spike is less than 0.1 → return "OFF"
    if np.mean(final_A_spike) < 0.1:
        return "OFF"
    #Check 3 — ON: more than 80% of cells in final_A_spike are above a_ss × 0.5 → return "ON"
    if np.mean(final_A_spike > 0.5 * a_ss) > 0.8:
        return "ON"
    #Check 4 — IRREGULAR: if none of the above → return "IRREGULAR"
    return "IRREGULAR"

# Main execution
if __name__ == "__main__":
    test_cases = [
        (8.0, 2.0),   # expect ON
        (5.0, 5.0),   # expect IRREGULAR
        (10.0, 8.0),   # expect TURING
        (3.0, 8.0),   # expect OFF
    ]
    for ba, bi in test_cases:
        regime = classify_regime(ba, bi)
        print(f"ba={ba}, bi={bi} → {regime}")
