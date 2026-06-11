from enum import Enum
from dataclasses import dataclass
from itertools import product
import networkx as nx
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Function that runs the whole pipeline from generating circuits to drawing the top Turing-positive ones
def run_pipeline():
    class Localization(Enum):
        INTRACELLULAR = "intracellular"
        MEMBRANE = "membrane"
        SECRETED = "secreted"

    # Define the Node class with a name and localization

    @dataclass
    class Node:
        name: str
        localization: Localization
    class Sign(Enum):
        ACTIVATION = "activation"
        INHIBITION = "inhibition"

    class Transport(Enum):
        CIS = "cis"
        TRANS_CONTACT = "trans_contact"
        TRANS_DIFFUSION = "trans_diffusion"

    # Define the Edge class with source and target nodes, sign, and transport

    @dataclass (frozen=True)
    class Edge:
        source: Node
        target: Node
        sign: Sign
        transport: Transport

    # The Enumerator class to generate all possible combinations of nodes and edges

    # Every possible node localization
    localizations = list(Localization)

    # Every possible edge states inculding absent
    edge_states = list(product(Sign, Transport)) + ["Absent"]

    # A itertools.product call that combines them &  generate every combination across all 4 slots at once
    def generate_circuits(localizations, edge_states):
        localizations_product = product(localizations, repeat=2)  # Generate all combinations of localizations for source and target nodes
        edge_states_product = product(edge_states, repeat=4)  # Generate all combinations of edge states for the 4 edges
        return list(product(localizations_product, edge_states_product))

    # Example usage
    circuits = generate_circuits(localizations, edge_states) # Generate all possible circuits ------ Side Note: This will create a large number of combinations
    print(f"Total number of circuits: {len(circuits)}")

    # Filters to ensure that the generated circuits are biologically plausible can be implemented here, such as ensuring that certain localizations are compatible with specific edge states.
    def is_valid_edge(edge, source_localization):
        if edge == "Absent":
            return True  # An absent edge is always valid
        if source_localization == Localization.SECRETED and edge[1] != Transport.TRANS_DIFFUSION:
            return False
        if source_localization == Localization.MEMBRANE and edge[1] != Transport.TRANS_CONTACT and edge[1] != Transport.CIS:
            return False
        if source_localization == Localization.INTRACELLULAR and edge[1] != Transport.CIS:
            return False
        return True
    # is_valid_circuit — a function that runs is_valid_edge on every edge in a circuit and returns False if any single edge fails.
    def is_valid_circuit(circuit):
        localizations, edges = circuit
        for i, edge in enumerate(edges):
            if i <= 1:
                source_localization = localizations[0]
            else:
                source_localization = localizations[1]
            if not is_valid_edge(edge, source_localization):
                return False
        return True
    # Filter the generated circuits to keep only valid ones
    def filter_circuits(circuits):
        return [circuit for circuit in circuits if is_valid_circuit(circuit)]   
    valid_circuits = filter_circuits(circuits)
    print(f"Total number of valid circuits: {len(valid_circuits)}")

    # Add node A with its correct localization attribute to a NetworkX graph
    edge_slots = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    graphs = []
    for circuit in valid_circuits:
        G = nx.DiGraph()
        node_locs, edges = circuit
        G.add_node("A", localization=node_locs[0])
        G.add_node("B", localization=node_locs[1])
        for i, edge in enumerate(edges):
            if edge == "Absent":
                continue
            source, target = edge_slots[i]
            sign = edge[0]
            transport = edge[1]
            G.add_edge(source, target, sign=sign, transport=transport)
        graphs.append(G)
    print(f"Total graphs built: {len(graphs)}")
    # A function called circuit_to_json that takes one valid circuit and its index number and returns a Python dictionary in that structure.
    def circuit_to_json(circuit, index):
        node_locs, edges = circuit
        edges_list = []
        for i, edge in enumerate(edges):
            if edge == "Absent":
                continue
            source, target = edge_slots[i]
            edges_list.append({
                "source": source,
                "target": target,
                "sign": edge[0].value,
                "transport": edge[1].value
            })
        return {
            "index": index,
            "nodes": {
                "A": node_locs[0].value,
                "B": node_locs[1].value
            },
            "edges": edges_list
        }
    json_circuits = [circuit_to_json(circuit, idx) for idx, circuit in enumerate(valid_circuits)]
    print(f"Total JSON circuits: {len(json_circuits)}")

    with open('valid_circuits.json', 'w') as f:
        json.dump(json_circuits, f, indent=2)
    print("Circuits saved to valid_circuits.json")

    # Example Usage: Small Query to find a specific circuit that matches the JAPI pattern
    def find_japi(circuits):
        for i, circuit in enumerate(circuits):
            nodes = circuit["nodes"]
            edges = circuit["edges"]
            if nodes["A"] == "membrane" and nodes["B"] == "secreted":
                has_activation = any(edge["source"] == "A" and edge["target"] == "A" and edge["sign"] == "activation" and edge["transport"] == "trans_contact" for edge in edges)
                has_inhibition = any(edge["source"] == "B" and edge["target"] == "A" and edge["sign"] == "inhibition" and edge["transport"] == "trans_diffusion" for edge in edges)
                if has_activation and has_inhibition and len(edges) == 2:
                    return i
        return -1

    japi_index = find_japi(json_circuits)
    if japi_index != -1:
        print(f"Found JAPI circuit at index: {japi_index}")
    else:
        print("No JAPI circuit found.")

    # Run this: git clone https://github.com/BenSwedlund/reaction_diffusion_simulations in terminal first, then run the following code to check if the JAPI circuit can produce Turing patterns based on the parameters defined in parameters.py


    from finding_steady_states import fast_stable_steady_state, hill_with_grads, _is_reaction_stable
    #build_jacobian that takes a_ss, i_ss, params, and k2 and returns J_spatial
    def build_jacobian(a_ss, i_ss, params, k2):
        _, dH_da, dH_di = hill_with_grads(
        a_ss, i_ss,
        params["act_half_sat"],
        params["inh_half_sat"],
        params["act_hill_coeff"],
        params["inh_hill_coeff"]
        )
        ba = params["act_prod_rate"]
        la = params["act_decay_rate"]
        bi = params["inh_prod_rate"]
        li = params["inh_decay_rate"]
        D_inh = params["inh_diffusion"]
        
        J_spatial = np.zeros((2, 2))
        # Jacobian based on the model equations
        J_spatial[0][0] = ba * dH_da - la - 0 * k2
        J_spatial[0][1] = ba * dH_di
        J_spatial[1][0] = bi * dH_da
        J_spatial[1][1] = bi * dH_di - li - D_inh * k2
        return J_spatial

    def check_turing(params):
        # Step 1 — finds steady state using Ben's function
        a_ss, i_ss, H_ss = fast_stable_steady_state(params)

        # Step 2 — checks stable without diffusion using Ben's function
        if not _is_reaction_stable(a_ss, i_ss, params):
            return False  # not even stable without diffusion, can't be Turing

        # Step 3 — checks if adding diffusion creates instability
        # tries a range of k² values from 0.01 to 10
        # for each k², builds J_spatial and checks if any eigenvalue is positive
        k_squared_values = np.linspace(0.01, 10, 100)
        for k2 in k_squared_values:
            J_spatial = build_jacobian(a_ss, i_ss, params, k2)
            eigenvalues = np.linalg.eigvals(J_spatial)
            if np.any(np.real(eigenvalues) > 0):
                return True  # found instability
        return False  # no instability found
    from parameters import params
    print(check_turing(params))

    # function called draw_circuit that takes one NetworkX graph and does just step 1 — draws the two nodes in the right positions with the right fill style.

    def draw_circuit(G, filename="circuit.png"):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # fixed positions for A and B
        positions = {
            "A": (0.3, 0.6),
            "B": (0.7, 0.6)
        }
        
        for node, (x, y) in positions.items():
            localization = G.nodes[node]["localization"]
            if localization == Localization.MEMBRANE:
                ax.add_patch(mpatches.Circle((x, y), radius=0.05, facecolor='white', edgecolor='black'))
            elif localization == Localization.SECRETED:
                ax.add_patch(mpatches.Circle((x, y), radius=0.05, facecolor='white', edgecolor='black'))
            elif localization == Localization.INTRACELLULAR:
                ax.add_patch(mpatches.Circle((x, y), radius=0.05, facecolor='black', edgecolor='black'))
            ax.text(x, y, node, ha='center', va='center', fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        r = 0.05  # circle radius

        for u, v, data in G.edges(data=True):
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            sign = data["sign"]
            transport = data["transport"]

            # line style based on transport
            if transport == Transport.CIS:
                linestyle = "solid"
                color = "black"
            elif transport == Transport.TRANS_CONTACT:
                linestyle = "dashed"
                color = "black"
            elif transport == Transport.TRANS_DIFFUSION:
                linestyle = "dashed"
                color = "blue"

            # arrowhead based on sign
            if sign == Sign.ACTIVATION:
                arrowstyle = "->"
            else:
                arrowstyle = "-["

            # self loop — curved arrow back to same node
            if u == v:
                ax.annotate("",
                    xy=(x1 + 0.04, y1 + 0.04),
                    xytext=(x1 - 0.04, y1 + 0.04),
                    arrowprops=dict(
                        arrowstyle=arrowstyle,
                        color=color,
                        linestyle=linestyle,
                        connectionstyle="arc3,rad=-0.8",
                        lw=1.5
                    ))
                continue

            # offset arrow to start and end at circle edge not center
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            x1_edge = x1 + r * dx / dist
            y1_edge = y1 + r * dy / dist
            x2_edge = x2 - r * dx / dist
            y2_edge = y2 - r * dy / dist

            ax.annotate("", xy=(x2_edge, y2_edge), xytext=(x1_edge, y1_edge),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    linestyle=linestyle,
                    edgecolor=color,
                    facecolor=color,
                    mutation_scale=15
                ))

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    japi_graph = graphs[japi_index]
    print(list(japi_graph.edges(data=True)))
    # Step 1: Generate all circuits
    circuits = generate_circuits(localizations, edge_states)
    
    # Step 2: Filter to valid ones
    valid_circuits = filter_circuits(circuits)
    
    # Step 3: Build NetworkX graphs
    graphs = []
    for circuit in valid_circuits:
        G = nx.DiGraph()
        node_locs, edges = circuit
        G.add_node("A", localization=node_locs[0])
        G.add_node("B", localization=node_locs[1])
        for i, edge in enumerate(edges):
            if edge == "Absent":
                continue
            source, target = edge_slots[i]
            sign = edge[0]
            transport = edge[1]
            G.add_edge(source, target, sign=sign, transport=transport)
        graphs.append(G)
    
    # Step 4: Save JSON library
    json_circuits = [circuit_to_json(circuit, idx) for idx, circuit in enumerate(valid_circuits)]
    with open('valid_circuits.json', 'w') as f:
        json.dump(json_circuits, f, indent=2)
    
    # Step 5: Run check_turing on each valid circuit and add a turing field to the JSON
    turing_results = []
    for idx, circuit in enumerate(valid_circuits):
        # Note: using JAPI parameters for all circuits
        # PI to confirm if circuit-specific parameters needed
        turing_results.append({
            "index": idx,
            "turing_positive": check_turing(params)
        })
    with open('turing_results.json', 'w') as f:
        json.dump(turing_results, f, indent=2)
    print("Pipeline complete. JSON files saved and top Turing-positive circuits drawn.")
    print(f"Total Turing-positive circuits: {sum(result['turing_positive'] for result in turing_results)}")
    # Step 6: Draw the top 5 Turing-positive circuits as diagrams
    top_turing_indices = [result["index"] for result in turing_results if result["turing_positive"]][:5]
    
    for i, idx in enumerate(top_turing_indices):
        draw_circuit(graphs[idx], filename=f"circuit_{i}.png")
run_pipeline()