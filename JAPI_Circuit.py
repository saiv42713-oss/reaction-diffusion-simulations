"""
JAPI_Circuit.py
Enumerates all 2-node gene circuits, filters to biologically valid ones,
checks Turing instability via LSA, and draws the top results.
"""

from enum import Enum
from dataclasses import dataclass
from itertools import product
import numpy as np
import networkx as nx
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from finding_steady_states import fast_stable_steady_state, _is_reaction_stable, hill_with_grads


class Localization(Enum):
    INTRACELLULAR = "intracellular"
    MEMBRANE = "membrane"
    SECRETED = "secreted"


class Sign(Enum):
    ACTIVATION = "activation"
    INHIBITION = "inhibition"


class Transport(Enum):
    CIS = "cis"
    TRANS_CONTACT = "trans_contact"
    TRANS_DIFFUSION = "trans_diffusion"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    sign: Sign
    transport: Transport


EDGE_SLOTS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
EDGE_STATES = list(product(Sign, Transport)) + ["Absent"]


def generate_circuits():
    loc_pairs = list(product(list(Localization), repeat=2))
    edge_combos = list(product(EDGE_STATES, repeat=4))
    return list(product(loc_pairs, edge_combos))


def is_valid_edge(edge, source_loc):
    if edge == "Absent":
        return True
    _, transport = edge
    if source_loc == Localization.SECRETED and transport != Transport.TRANS_DIFFUSION:
        return False
    if source_loc == Localization.MEMBRANE and transport not in (Transport.TRANS_CONTACT, Transport.CIS):
        return False
    if source_loc == Localization.INTRACELLULAR and transport != Transport.CIS:
        return False
    return True


def is_valid_circuit(circuit):
    locs, edges = circuit
    for i, edge in enumerate(edges):
        src_loc = locs[0] if i <= 1 else locs[1]
        if not is_valid_edge(edge, src_loc):
            return False
    return True


def filter_circuits(circuits):
    return [c for c in circuits if is_valid_circuit(c)]


def build_graph(circuit):
    G = nx.DiGraph()
    locs, edges = circuit
    G.add_node("A", localization=locs[0])
    G.add_node("B", localization=locs[1])
    for i, edge in enumerate(edges):
        if edge == "Absent":
            continue
        src, tgt = EDGE_SLOTS[i]
        sign, transport = edge
        e = Edge(source=src, target=tgt, sign=sign, transport=transport)
        G.add_edge(src, tgt, edge=e, sign=e.sign, transport=e.transport)
    return G


def circuit_to_json(circuit, index):
    locs, edges = circuit
    edge_list = []
    for i, edge in enumerate(edges):
        if edge == "Absent":
            continue
        src, tgt = EDGE_SLOTS[i]
        sign, transport = edge
        e = Edge(source=src, target=tgt, sign=sign, transport=transport)
        edge_list.append({
            "source": e.source,
            "target": e.target,
            "sign": e.sign.value,
            "transport": e.transport.value,
        })
    return {
        "index": index,
        "nodes": {"A": locs[0].value, "B": locs[1].value},
        "edges": edge_list,
    }


def find_japi(json_circuits):
    for i, c in enumerate(json_circuits):
        if c["nodes"]["A"] != "membrane" or c["nodes"]["B"] != "secreted":
            continue
        has_act = any(
            e["source"] == "A" and e["target"] == "A"
            and e["sign"] == "activation" and e["transport"] == "trans_contact"
            for e in c["edges"]
        )
        has_inh = any(
            e["source"] == "B" and e["target"] == "A"
            and e["sign"] == "inhibition" and e["transport"] == "trans_diffusion"
            for e in c["edges"]
        )
        if has_act and has_inh and len(c["edges"]) == 2:
            return i
    return -1


def _diffusion_for_node(node, circuit, params):
    """Returns the spatial diffusion coefficient for a node.
    (A node only diffuses if it has at least one TRANS_DIFFUSION outgoing edge.)"""
    _, edges = circuit
    for i, edge in enumerate(edges):
        if edge == "Absent":
            continue
        src, _ = EDGE_SLOTS[i]
        _, transport = edge
        if src == node and transport == Transport.TRANS_DIFFUSION:
            key = "act_diffusion" if node == "A" else "inh_diffusion"
            return params.get(key, 0.0)
    return 0.0


def _interaction_sign(src, tgt, circuit):
    """Returns +1 for activation, -1 for inhibition, 0 if the edge is absent."""
    _, edges = circuit
    for i, edge in enumerate(edges):
        if edge == "Absent":
            continue
        s, t = EDGE_SLOTS[i]
        if s == src and t == tgt:
            sign, _ = edge
            return 1 if sign == Sign.ACTIVATION else -1
    return 0


def build_jacobian(circuit, a_ss, i_ss, params, k2):
    """
    Builds the 2x2 spatial Jacobian for a specific circuit at wavenumber k2.

    Each circuit determines:
      - which interactions exist (A->A, A->B, B->A, B->B) and their signs
      - which species diffuse, derived from their transport modes

    Species order: A=row/col 0, B=row/col 1.
    """
    _, dH_da, dH_di = hill_with_grads(
        a_ss, i_ss,
        params["act_half_sat"], params["inh_half_sat"],
        params["act_hill_coeff"], params["inh_hill_coeff"],
    )
    ba = params["act_prod_rate"]
    la = params["act_decay_rate"]
    bi = params["inh_prod_rate"]
    li = params["inh_decay_rate"]
    Da = _diffusion_for_node("A", circuit, params)
    Di = _diffusion_for_node("B", circuit, params)

    s_AA = _interaction_sign("A", "A", circuit)
    s_BA = _interaction_sign("B", "A", circuit)
    s_AB = _interaction_sign("A", "B", circuit)
    s_BB = _interaction_sign("B", "B", circuit)

    J = np.zeros((2, 2))
    J[0, 0] = ba * dH_da * s_AA - la - Da * k2
    J[0, 1] = ba * dH_di * s_BA
    J[1, 0] = bi * dH_da * s_AB
    J[1, 1] = bi * dH_di * s_BB - li - Di * k2
    return J


def check_turing(circuit, params):
    """Return True if this specific circuit satisfies the Turing instability condition."""
    a_ss, i_ss, _ = fast_stable_steady_state(params)
    if not _is_reaction_stable(a_ss, i_ss, params):
        return False
    for k2 in np.linspace(0.01, 10, 100):
        J = build_jacobian(circuit, a_ss, i_ss, params, k2)
        if np.any(np.real(np.linalg.eigvals(J)) > 0):
            return True
    return False


NODE_STYLE = {
    Localization.MEMBRANE: dict(facecolor="white", edgecolor="black", linewidth=2),
    Localization.SECRETED: dict(facecolor="#d9d9d9", edgecolor="black", linewidth=2),
    Localization.INTRACELLULAR: dict(facecolor="black", edgecolor="black", linewidth=2),
}
EDGE_COLOR = {
    Transport.CIS: "black",
    Transport.TRANS_CONTACT: "black",
    Transport.TRANS_DIFFUSION: "#1f78b4",
}
EDGE_LINE = {
    Transport.CIS: "solid",
    Transport.TRANS_CONTACT: "dashed",
    Transport.TRANS_DIFFUSION: "dashed",
}


def draw_circuit(G, filename="circuit.png", title=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = {"A": (0.3, 0.6), "B": (0.7, 0.6)}
    r = 0.055

    for node, (x, y) in pos.items():
        loc = G.nodes[node]["localization"]
        ax.add_patch(mpatches.Circle((x, y), radius=r, **NODE_STYLE[loc], zorder=3))
        ax.text(x, y, node, ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

    for u, v, data in G.edges(data=True):
        e = data["edge"]
        color = EDGE_COLOR[e.transport]
        lstyle = EDGE_LINE[e.transport]
        astyle = "->" if e.sign == Sign.ACTIVATION else "-["
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if u == v:
            ax.annotate("",
                xy=(x1 + 0.05, y1 + 0.05), xytext=(x1 - 0.05, y1 + 0.05),
                arrowprops=dict(arrowstyle=astyle, color=color,
                                linestyle=lstyle,
                                connectionstyle="arc3,rad=-0.8", lw=1.8))
        else:
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy)
            ax.annotate("",
                xy=(x2 - r * dx / dist, y2 - r * dy / dist),
                xytext=(x1 + r * dx / dist, y1 + r * dy / dist),
                arrowprops=dict(arrowstyle=astyle, linestyle=lstyle,
                                edgecolor=color, facecolor=color,
                                mutation_scale=16, lw=1.8))

    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Membrane (A)"),
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="black", label="Secreted (B)"),
        mpatches.Patch(facecolor="black", edgecolor="black", label="Intracellular"),
        plt.Line2D([0], [0], color="black", lw=1.8, linestyle="solid", label="Cis"),
        plt.Line2D([0], [0], color="black", lw=1.8, linestyle="dashed", label="Trans-contact"),
        plt.Line2D([0], [0], color="#1f78b4", lw=1.8, linestyle="dashed", label="Trans-diffusion"),
        plt.Line2D([0], [0], color="black", lw=0, marker=">", label="Activation →"),
        plt.Line2D([0], [0], color="black", lw=0, marker="$⊣$", markersize=10, label="Inhibition ⊣"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              fontsize=7.5, ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.3, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def run_pipeline():
    from parameters import params

    print("Generating circuits...")
    all_circuits = generate_circuits()
    print(f"  Total: {len(all_circuits)}")
    valid = filter_circuits(all_circuits)
    print(f"  Valid: {len(valid)}")

    graphs = [build_graph(c) for c in valid]
    print(f"  Graphs built: {len(graphs)}")

    json_circuits = [circuit_to_json(c, i) for i, c in enumerate(valid)]
    with open("valid_circuits.json", "w") as f:
        json.dump(json_circuits, f, indent=2)
    print("  Saved valid_circuits.json")

    japi_idx = find_japi(json_circuits)
    if japi_idx >= 0:
        print(f"  JAPI circuit at index {japi_idx}")
        draw_circuit(graphs[japi_idx], "japi_circuit.png", title="JAPI Circuit")
    else:
        print("  JAPI not found.")

    print("Running turing LSA check...")
    turing_results = [
        {"index": i, "turing_positive": check_turing(circuit, params)}
        for i, circuit in enumerate(valid)
    ]
    with open("turing_results.json", "w") as f:
        json.dump(turing_results, f, indent=2)
    n_turing = sum(r["turing_positive"] for r in turing_results)
    print(f"  Turing-positive: {n_turing} / {len(valid)}")

    top5 = [r["index"] for r in turing_results if r["turing_positive"]][:5]
    for rank, idx in enumerate(top5):
        draw_circuit(graphs[idx], f"circuit_top{rank+1}.png",
                     title=f"Turing circuit #{rank+1}  (index {idx})")
    print("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()