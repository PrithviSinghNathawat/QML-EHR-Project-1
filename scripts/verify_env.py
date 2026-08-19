import sys
import pennylane as qml
from pennylane import numpy as pnp

print("python:", sys.version)
print("pennylane:", qml.version())

n_qubits = 6
dev = qml.device("lightning.qubit", wires=n_qubits)

device_name = getattr(dev, "name", type(dev).__name__)
device_class = type(dev).__module__ + "." + type(dev).__name__
print("device name:", device_name)
print("device class:", device_class)
assert "lightning" in device_class.lower(), (
    f"device silently fell back to a non-lightning backend: {device_class}"
)

@qml.qnode(dev, diff_method="adjoint")
def circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for l in range(weights.shape[0]):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

x = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=False)
weights = pnp.array(
    [[0.1 * (i + j) for j in range(n_qubits)] for i in range(3)],
    requires_grad=True,
)

out = circuit(x, weights)
print("forward output:", out)

grad_fn = qml.grad(circuit)
grad = grad_fn(x, weights)
assert grad.shape == weights.shape
assert not pnp.allclose(grad, 0.0), "adjoint gradient is all zero"
print("adjoint gradient OK, shape:", grad.shape)

print("\nENV CHECK: OK")
