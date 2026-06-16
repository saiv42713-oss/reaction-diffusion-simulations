import numpy as np
from finding_steady_states import fast_stable_steady_state, find_unstable_fixed_point
from simulation_2D import run_coupled_hex
from radial_autocor_npz import analyze_image_autocorrelation


SIM_PARAMS = {
    "Ny": 100,
    "Nx": 100,
    "steps": 50000,
    "dt": 0.01,
    "dx": 1.0,
    "stopping_threshold": 1e-4,
    "min_steps": 5000,
    "save_every": 1000,
    "spike_value": 2.0,
    "n_points": 10
}

BASE_PARAMS = {
    "act_half_sat": 1.0,
    "inh_half_sat": 1.0,
    "act_decay_rate": 1.0,
    "inh_decay_rate": 0.5,
    "act_diffusion": 0.0,
    "inh_diffusion": 10.0,
    "basal_prod": 0.0,
    "act_hill_coeff": 3,
    "inh_hill_coeff": 3,
    "act_prod_rate": 5.0,
}


def classify_regime(act_prod_rate, inh_prod_rate, debug=False):
    np.random.seed(42)
    params = {**BASE_PARAMS,
              "act_prod_rate": act_prod_rate,
              "inh_prod_rate": inh_prod_rate}

    a_ss, i_ss, _ = fast_stable_steady_state(params)
    # Although, i_ss, final_step, and R_hist, are not used, they are returned by the function and are therefore expected by, annoyingly. 
    # Also, a_ss is not there because it being returned as a tuple not float

    if a_ss == 0.0:
        spike_value_first = SIM_PARAMS["spike_value"]
        spike_value_second = SIM_PARAMS["spike_value"] * 3
        init_mode_first = "random_uniform_over0"
        init_mode_second = "activator_random_spikes"
    else:
        spike_value_first = a_ss
        spike_value_second = a_ss
        init_mode_first = "noise_around_state"
        init_mode_second = "noise_around_state"
    # If a_ss is 0.0 then that means off because that is natural resting state
    #function that produces an fast_unstable_state function. So to prevent Im replacing the a_ss with the spike_value, to reduce probabillity of missing turing/irregular

    print(f"a_ss before threshold: {a_ss}")
    threshold = find_unstable_fixed_point(params, a_ss, i_ss, activator_type="juxtacrine")
    
    ref = a_ss if a_ss > 0.0 else spike_value_first

    if threshold is not None:
        init_amplitude_first = threshold * 0.1
        init_amplitude_second = threshold * 1.5
    else:
        init_amplitude_first = ref / 5
        init_amplitude_second = ref * 2
    
        
    print(f"threshold={threshold}, amp1={init_amplitude_first}, amp2={init_amplitude_second}")    

    A_hist, R_hist, final_step, _, _ = run_coupled_hex(
        SIM_PARAMS["Ny"], SIM_PARAMS["Nx"],
        SIM_PARAMS["steps"], SIM_PARAMS["dt"], SIM_PARAMS["dx"],
        params,
        SIM_PARAMS["stopping_threshold"],
        SIM_PARAMS["min_steps"],
        init_mode=init_mode_first,
        activator_type="juxtacrine",
        spike_value=spike_value_first,
        save_every=SIM_PARAMS["save_every"],
        nucleation_rate=0,
        init_amplitude=init_amplitude_first
    )
    final_A = A_hist[-1]
    # Same for this.
    A_hist, R_hist, final_step, _, _ = run_coupled_hex(
        SIM_PARAMS["Ny"], SIM_PARAMS["Nx"],
        SIM_PARAMS["steps"], SIM_PARAMS["dt"], SIM_PARAMS["dx"],
        params,
        SIM_PARAMS["stopping_threshold"],
        SIM_PARAMS["min_steps"],
        init_mode=init_mode_second,
        activator_type="juxtacrine",
        spike_value=spike_value_second,
        save_every=SIM_PARAMS["save_every"],
        nucleation_rate=0,
        init_amplitude=init_amplitude_second,
        n_points=SIM_PARAMS["n_points"],
    )
    final_A_spike = A_hist[-1]

    # Check 1 - uniform ON (nucleation sim)
    if debug:
        fft_nuc = np.abs(np.fft.fft2(final_A))
        fft_nuc[0, 0] = 0
        # fft_ratio_nuc = np.max(fft_nuc) / (np.mean(fft_nuc) + 1e-10): Just incase we want to use it for debugging later, but not currently used in the classification.
    cv = np.std(final_A) / (np.mean(final_A) + 1e-10)  # coefficient of variation
    if (np.mean(final_A) > 0.8 * ref and cv < 0.05):
        return "ON"


    # Check 2 - spike ON
    if np.mean(final_A_spike > 0.5 * ref) > 0.8:
        return "ON"
    
    # DEBUG
    if debug:
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
        fft2 = np.abs(np.fft.fft2(final_A_spike))
        fft2[0, 0] = 0
        print(f"FFT2 peak: {np.max(fft2):.2f}, mean: {np.mean(fft2):.2f}, ratio: {np.max(fft2)/np.mean(fft2):.2f}")

    # Check 3 — TURING/IRREGULAR
    if np.mean(final_A) > 0.1 and np.std(final_A) > 0.1:
        A_2d, df = analyze_image_autocorrelation(final_A)
        radial_ac = df["radial_autocorrelation"].values
        if np.any(radial_ac < 0):
            fft = np.abs(np.fft.fft2(final_A))
            fft[0, 0] = 0
            peak = np.max(fft)
            mean_fft = np.mean(fft)
            if peak / mean_fft > 35:
                return "TURING"
            else:
                return "IRREGULAR"
            
    # Check 3b — IRREGULAR from spike simulation
    if np.mean(final_A_spike) > 0.01 and np.std(final_A_spike) > 0.01:
        A_2d, df = analyze_image_autocorrelation(final_A_spike)
        radial_ac = df["radial_autocorrelation"].values
        if np.any(radial_ac < 0):
            fft = np.abs(np.fft.fft2(final_A_spike))
            fft[0, 0] = 0
            peak = np.max(fft)
            mean_fft = np.mean(fft)
            if peak / mean_fft > 35:
                return "TURING"
            else:
                return "IRREGULAR"


    # Check 4 — OFF
    if np.mean(final_A_spike) < 0.1:
        return "OFF"

    # Check 5 — fallback
    return "IRREGULAR"


def classify_regime_and_return_fields(act_prod_rate, inh_prod_rate):
    params = {**BASE_PARAMS,
              "act_prod_rate": act_prod_rate,
              "inh_prod_rate": inh_prod_rate}
    a_ss, _, _ = fast_stable_steady_state(params)
    A_hist, R_hist, _, _, _ = run_coupled_hex(
        SIM_PARAMS["Ny"], SIM_PARAMS["Nx"],
        SIM_PARAMS["steps"], SIM_PARAMS["dt"], SIM_PARAMS["dx"],
        params,
        SIM_PARAMS["stopping_threshold"],
        SIM_PARAMS["min_steps"],
        init_mode="noise_around_state",
        activator_type="juxtacrine",
        spike_value=a_ss,
        save_every=SIM_PARAMS["save_every"]
    )
    return A_hist[-1], R_hist[-1]


if __name__ == "__main__":
    test_cases = [
        (5.0, 1.0),   # → ON
        (5.0, 5.0),   # → TURING
        (5.0, 12.0),  # → IRREGULAR
        (5.0, 14.0), # → OFF
    ]
    for ba, bi in test_cases:
        regime = classify_regime(ba, bi)
        print(f"ba={ba}, bi={bi} → {regime}")

params = {**BASE_PARAMS, "act_prod_rate": 5.0, "inh_prod_rate": 25.0}
a_ss, i_ss, _ = fast_stable_steady_state(params)
print(f"a_ss={a_ss}, i_ss={i_ss}")

params = {**BASE_PARAMS, "act_prod_rate": 5.0, "inh_prod_rate": 25.0}

H_lo, H_hi = 1e-9, 1.0 - 1e-9
Hs = np.linspace(H_lo, H_hi, 64)

A = params["act_prod_rate"] / params["act_decay_rate"]
I = params["inh_prod_rate"] / params["inh_decay_rate"]

from finding_steady_states import hill_with_grads
gs = []
for H in Hs:
    Hval, _, _ = hill_with_grads(A*H, I*H, 1.0, 1.0, 10, 4, 0.0)
    gs.append(Hval - H)

print(list(zip(Hs.round(4), [round(g,6) for g in gs])))

regime = classify_regime(5.0, 12.0, debug=True)
print(f"ba=5.0, bi=12.0 → {regime}")
