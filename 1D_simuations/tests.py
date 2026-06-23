from parameters import params, N, steps, dt, dx, save_every, spike_value, stopping_threshold
from simulation import run_coupled_neumann
from visualize import animate_histories
import argparse


def test_inhibitor_diffusion_only():
    """
    Pure diffusion test: inhibitor spike with no production/decay.
    Expect flattening over time without vanishing.
    """
    p = params.copy()
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0
    p["act_prod_rate"] = 0.0
    p["act_decay_rate"] = 0.0

    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="both_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: Inhibitor diffusion only")
    animate_histories(A_hist, R_hist, save_every, title="Inhibitor diffusion only (Neumann)")


def test_activator_and_inhibitor_diffusion():
    """
    Pure decay test: activator spike with no production.
    Expect exponential decay toward zero.
    """
    p = params.copy()
    p["act_prod_rate"] = 0.0
    p["inh_prod_rate"] = 0.0
    p["inh_decay_rate"] = 0.0
    p["act_decay_rate"] = 0.0

    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="both_spike",
        activator_type="soluble",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Test: Activator and Inhibitor diffusion (membrane)")
    animate_histories(A_hist, R_hist, save_every, title="Activator & Inhibitor diffusion (Neumann)")


def test_decay_only():
    """
    Pure decay test: activator spike with no production.
    Expect exponential decay toward zero.
    """
    p = params.copy()
    p["act_prod_rate"] = 0.0
    p["inh_prod_rate"] = 0.0

    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="both_spike",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Testing: decay only")
    animate_histories(A_hist, R_hist, save_every, title="Decay only (Neumann)")

def test_activator_propagation_only_no_diffusion():
    """
    Propagation test: activator spike with production and decay, but no inhibitor production.
    Expect wave propagation toward steady-state value.
    """
    p = params.copy()
    p["act_prod_rate"] = 3.0
    p["inh_prod_rate"] = 0.0

    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        activator_type="membrane-tethered",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Testing: Membrane-tethered activator signal propagation")
    animate_histories(A_hist, R_hist, save_every, title="Signal Propagation, no diffusion (Neumann)")

def test_activator_propagation_only_with_diffusion():
    """
    Propagation test: activator spike with production and decay, but no inhibitor production.
    Expect wave propagation toward steady-state value.
    """
    p = params.copy()
    p["act_prod_rate"] = 3.0
    p["inh_prod_rate"] = 0.0

    A_hist, R_hist, final_step = run_coupled_neumann(
        N, steps, dt, dx, p, stopping_threshold,
        init_mode="activator_spike",
        activator_type="soluble",
        spike_value=spike_value,
        save_every=save_every,
    )

    print("Testing: diffusible activator signal propagation")
    animate_histories(A_hist, R_hist, save_every, title="Signal Propagation, with diffusion (Neumann)")


def main():
    tests = {
        "inhibitor_diffusion_only": test_inhibitor_diffusion_only,
        "activator_and_inhibitor_diffusion": test_activator_and_inhibitor_diffusion,
        "decay_only": test_decay_only,
        "activator_propagation_no_diffusion": test_activator_propagation_only_no_diffusion,
        "activator_propagation_with_diffusion": test_activator_propagation_only_with_diffusion,
    }

    parser = argparse.ArgumentParser(description="Run specific test cases.")
    parser.add_argument("test", choices=tests.keys(), help="Choose which test to run")
    args = parser.parse_args()

    tests[args.test]()  # directly call the mapped function

if __name__ == "__main__":
    main()
