"""Render the 6-qubit, 3-layer VQC circuit diagram used in the timing spike
(and to be reused by Arm 4/5) as text, for the review presentation.
"""
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 6
N_LAYERS = 3

dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev, diff_method="adjoint")
def circuit(x, weights):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))


x = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
weights = pnp.array([[0.1 * (i + j) for j in range(N_QUBITS)] for i in range(N_LAYERS)])

diagram = qml.draw(circuit, max_length=200)(x, weights)

with open("docs/circuit_diagram.txt", "w", encoding="utf-8") as f:
    f.write(f"6-qubit, {N_LAYERS}-layer VQC: angle encoding (RY) + RY ansatz + linear CNOT entangling\n")
    f.write("=" * 80 + "\n\n")
    f.write(diagram + "\n")

print(diagram.encode("ascii", "replace").decode("ascii"))
print("\nwritten to docs/circuit_diagram.txt")

fig, ax = qml.draw_mpl(circuit, style="pennylane")(x, weights)
fig.savefig("docs/circuit_diagram.png", dpi=220, bbox_inches="tight")
print("written to docs/circuit_diagram.png (dpi=220)")
