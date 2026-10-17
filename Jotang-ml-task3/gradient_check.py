from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "gradient_comparison.txt"
np.set_printoptions(precision=10, suppress=True, linewidth=110)


def compare(name, manual, automatic, atol=1e-12, rtol=1e-10):
    """Show shape, value, and maximum absolute error; fail on disagreement."""
    manual = np.asarray(manual, dtype=np.float64)
    automatic = np.asarray(automatic, dtype=np.float64)
    assert manual.shape == automatic.shape, (name, manual.shape, automatic.shape)
    error = float(np.max(np.abs(manual - automatic)))
    print(f"{name}: shape={manual.shape}, max_abs_error={error:.3e}")
    print(f"  NumPy   = {np.array2string(manual)}")
    print(f"  PyTorch = {np.array2string(automatic)}")
    assert np.allclose(manual, automatic, atol=atol, rtol=rtol), name
    return error


def simple_chain():
    """f=(xy+x²)²; all nodes are scalar (shape ())."""
    x = np.array(1.5, dtype=np.float64)
    y = np.array(-0.5, dtype=np.float64)

    # Forward graph: x,y -> multiply; x -> square; both -> add -> outer square.
    product = np.multiply(x, y)
    x_squared = np.square(x)
    sum_ = np.add(product, x_squared)
    f = np.square(sum_)

    # Backward graph: gradients of f with respect to each saved node.
    df_dsum = 2 * sum_
    df_dproduct = df_dsum * 1
    df_dx_squared = df_dsum * 1
    df_dx = df_dproduct * y + df_dx_squared * (2 * x)  # add both paths from x
    df_dy = df_dproduct * x

    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    f_t = (x_t * y_t + x_t**2) ** 2
    f_t.backward()

    print("=== Practice 1: f(x,y)=(xy+x^2)^2 ===")
    print(f"x={x}, y={y}; xy={product}, x^2={x_squared}, sum={sum_}, f={f}")
    print(f"backward: df/dsum={df_dsum}, df/d(xy)={df_dproduct}, df/d(x^2)={df_dx_squared}")
    compare("f", f, f_t.detach().numpy())
    ex = compare("df/dx", df_dx, x_t.grad.numpy())
    ey = compare("df/dy", df_dy, y_t.grad.numpy())
    print(f"Result: {'PASS' if max(ex, ey) < 1e-12 else 'FAIL'}\n")


def network_numpy(X, Y, W1, b1, W2, b2):
    """Shapes: X(n,2), Y(n,1), W1(2,3), b1(3,), W2(3,1), b2(1,)."""
    n = X.shape[0]
    # Forward: Z1/A1 (n,3), Z2/P (n,1), loss scalar ().
    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    P = 1.0 / (1.0 + np.exp(-Z2))
    loss = -np.mean(Y * np.log(P) + (1.0 - Y) * np.log(1.0 - P))

    # Reverse chain. The mean over n samples contributes a factor of 1/n.
    dP = (-(Y / P) + (1.0 - Y) / (1.0 - P)) / n  # (n,1)
    dZ2 = dP * P * (1.0 - P)  # (n,1); equals (P-Y)/n
    assert np.allclose(dZ2, (P - Y) / n, atol=1e-14)
    dW2 = A1.T @ dZ2  # (3,1)
    db2 = dZ2.sum(axis=0)  # (1,); undo bias broadcasting
    dA1 = dZ2 @ W2.T  # (n,3)
    dZ1 = dA1 * (Z1 > 0.0)  # (n,3); ReLU'(0)=0, same as PyTorch
    dW1 = X.T @ dZ1  # (2,3)
    db1 = dZ1.sum(axis=0)  # (3,); undo bias broadcasting
    dX = dZ1 @ W1.T  # (n,2)

    values = {"X": X, "Y": Y, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P, "loss": loss}
    grads = {"X": dX, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
             "Z1": dZ1, "A1": dA1, "Z2": dZ2, "P": dP}
    return values, grads


def network_torch(X, Y, W1, b1, W2, b2):
    """Same numbers and float64 dtype; retain intermediate gradients for diagnosis."""
    t = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)
    x, w1, bias1, w2, bias2 = map(t, (X, W1, b1, W2, b2))
    target = torch.tensor(Y, dtype=torch.float64)
    z1 = x @ w1 + bias1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + bias2
    p = torch.sigmoid(z2)
    for node in (z1, a1, z2, p):
        node.retain_grad()
    loss = F.binary_cross_entropy(p, target, reduction="mean")
    loss.backward()
    values = {"X": x, "Y": target, "Z1": z1, "A1": a1, "Z2": z2, "P": p, "loss": loss}
    grads = {"X": x.grad, "W1": w1.grad, "b1": bias1.grad, "W2": w2.grad,
             "b2": bias2.grad, "Z1": z1.grad, "A1": a1.grad, "Z2": z2.grad, "P": p.grad}
    return values, grads


def network_check(indices):
    # Exactly the Task 1 make_moons generation; use chosen original row indices.
    X_all, y_all = make_moons(n_samples=1000, noise=0.2, random_state=4)
    X = np.asarray(X_all[indices], dtype=np.float64)  # (n,2)
    Y = np.asarray(y_all[indices, None], dtype=np.float64)  # (n,1)
    W1 = np.array([[0.7, -0.4, 0.3], [-0.2, 0.5, 0.8]], dtype=np.float64)  # (2,3)
    b1 = np.array([0.1, -0.1, -0.2], dtype=np.float64)  # (3,)
    W2 = np.array([[0.6], [-0.7], [0.4]], dtype=np.float64)  # (3,1)
    b2 = np.array([-0.05], dtype=np.float64)  # (1,)

    print(f"=== Practice 2: n={len(indices)}, original row indices={indices} ===")
    print(f"X shape={X.shape}, X={np.array2string(X)}")
    print(f"Y shape={Y.shape}, Y={np.array2string(Y)}")
    print(f"W1 shape={W1.shape}, W1={np.array2string(W1)}")
    print(f"b1 shape={b1.shape}, b1={np.array2string(b1)}")
    print(f"W2 shape={W2.shape}, W2={np.array2string(W2)}")
    print(f"b2 shape={b2.shape}, b2={np.array2string(b2)}")

    manual_values, manual_grads = network_numpy(X, Y, W1, b1, W2, b2)
    torch_values, torch_grads = network_torch(X, Y, W1, b1, W2, b2)
    for name in ("Z1", "A1", "Z2", "P", "loss"):
        tensor = torch_values[name]
        compare(name, manual_values[name], tensor.detach().numpy())
    max_parameter_error = 0.0
    for name in ("W1", "b1", "W2", "b2"):
        max_parameter_error = max(max_parameter_error,
                                  compare("d" + name, manual_grads[name], torch_grads[name].numpy()))
    for name in ("X", "Z1", "A1", "Z2", "P"):
        compare("d" + name, manual_grads[name], torch_grads[name].numpy())
    print(f"Parameter gradient max absolute error: {max_parameter_error:.3e}")
    print("Result: PASS (all values and gradients agree)\n")


def main():
    transcript = StringIO()
    with redirect_stdout(transcript):
        simple_chain()
        network_check([0])
        network_check([0, 1, 9, 11])
    result = transcript.getvalue().rstrip() + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(result, encoding="utf-8")
    print(result, end="")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
