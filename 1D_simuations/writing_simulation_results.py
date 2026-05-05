import os

def str2bool(v):
    """Helper function to parse True/False command line args."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

def write_simulation_results(args, activator_type, init_mode, spike_value, params, A_hist, R_hist, final_step):
    """Write simulation variables and last frame data to TXT and return output paths."""
    outdir = "simulation_results"
    os.makedirs(outdir, exist_ok=True)

    outfile_txt = os.path.join(outdir, args.output + ".txt")
    outfile_png = os.path.join(outdir, args.output + "_end.png")

    # Save results as text
    with open(outfile_txt, "w") as f:
        # basic variables
        f.write(f"activator_type\t{activator_type}\n")
        f.write(f"init_mode\t{init_mode}\n")
        f.write(f"spike_value\t{spike_value}\n")
        f.write(f"total_steps\t{final_step}\n")

        # params dictionary
        for k, v in params.items():
            f.write(f"{k}\t{v}\n")

        # last element of A_hist and R_hist
        if A_hist:
            f.write("A_hist_last\t" + "\t".join(map(str, A_hist[-1])) + "\n")
        if R_hist:
            f.write("R_hist_last\t" + "\t".join(map(str, R_hist[-1])) + "\n")

    print(f"Results saved to {outfile_txt}")
    return outfile_txt, outfile_png
