# Spread Option Valuation Benchmark (Hurd–Zhou Fourier Formula)

This repository implements a reproducible benchmark for **European spread option** valuation using the Hurd–Zhou Fourier representation. The codebase provides:

* a numerically stable implementation of the **payoff transform**,
* a **model-agnostic** interface for joint characteristic functions,
* an **adaptive Gauss–Legendre** quadrature with explicit error control,
* tooling for **benchmarking across strikes and models**.

---

## Background and Notation

Consider the spread payoff $\max(S_1 - \eta S_2 - K, 0)$ with $S_i = e^{X_i}$. With the Hurd–Zhou payoff transform $\widehat{P}(u_1,u_2)$ and a model’s joint characteristic function $\phi(u_1,u_2;T)$, the price admits the Fourier representation

$$
\mathrm{Spr}(X_0;T)
= \frac{e^{-rT}K}{(2\pi)^2}
\iint_{\mathbb{R}^2}
e^{i(u_1 b_1 + u_2 b_2)} \,
\phi(u_1,u_2;T)\,
\widehat{P}(u_1,u_2)\,du_1\,du_2,
$$

where $b_1=\log(S_1/K)$ and $b_2=\log(\eta S_2/K)$. A vertical shift $u \mapsto u + i\omega$ (with $\omega\in\mathbb{R}^2$) is employed to ensure absolute integrability and numerical stability.

The payoff transform is computed via stable log-gamma evaluations:

$$
\widehat{P}(u_1,u_2)
= \frac{\Gamma(i(u_1+u_2)-1)\,\Gamma(-iu_2)}{\Gamma(1+iu_1)}.
$$

---

## Numerical Method

### Quadrature and Truncation

The integral is evaluated on a square box $[-U,U]^2$ using tensorized **Gauss–Legendre** quadrature of order $n$ per axis. Two numerical errors arise:

* **Discretization (quadrature) error** on $[-U,U]^2$,
* **Tail (truncation) error** from restricting $\mathbb{R}^2$ to $[-U,U]^2$.

Floating-point roundoff is mitigated using **Kahan compensated summation** on the real-valued weighted integrand.

### Effective Spacing and Oscillation Control

Although Gauss–Legendre nodes are nonuniform, we define an **effective spacing**

$$
h \approx \frac{2U}{n}
$$

as a proxy for node density. The oscillatory factor $e^{i(u_1b_1+u_2b_2)}$ motivates a target spacing

$$
h_{\text{tgt}} = \min\!\left(\frac{\pi}{12\,B},\; h_{\max}\right), \quad
B = \max\{|b_1|,\;|b_2|,\;|b_1+b_2|\}.
$$

This ensures adequate resolution of the dominant oscillations while allowing $U$ to grow without degrading the mesh density (we scale $n$ with $U$ so that $2U/n \approx h_{\text{tgt}}$).

---

## Error Tolerance and Adaptive Procedure

Let `tol` denote the target absolute error. We enforce

$$
\text{discretization error} \;+\; \text{tail error} \;\lesssim\; \texttt{tol}
$$

via two nested controls with a budget split (default $25\%$ for discretization, $75\%$ for tail).

### Discretization Control (Fixed $U$)

At a fixed $U$, compute quadrature with $n$ and $2n$ nodes:

$$
v(n),\; v(2n).
$$

Refine $n \mapsto 2n$ until

$$
\lvert v(2n) - v(n) \rvert \le \mathrm{disc}_{\mathrm{budget}} = 0.25\cdot \mathrm{tol}.
$$

Since $v(2n)-v(n)=E_n-E_{2n}$ and $E_{2n}\ll E_n$ for smooth integrands, this difference is a reliable proxy for the discretization error on $[-U,U]^2$.

### Tail Control (Increasing $U$ at Nearly Constant $h$)

Having fixed the discretization at $U$, enlarge the box $U \leftarrow \rho U$ (default $\rho=1.5$) and scale $n$ so that $2U/n\approx h_{\mathrm{tgt}}$. Let

$$
v(U_{\mathrm{old}}),\quad v(U_{\mathrm{new}}).
$$

Iterate until

$$
\lvert v(U_{\mathrm{new}}) - v(U_{\mathrm{old}}) \rvert \le \mathrm{tail}_{\mathrm{budget}} = 0.75\cdot \mathrm{tol}.
$$

Maintaining $h$ approximately constant isolates the tail contribution; the discretization level remains comparable across successive boxes.

## Usage

### Installation

```bash
pip install -e .
```

### Example

```python
import time
import numpy as np
from spread_benchmark import AlphaVGModel, SpreadPricer

# Model (α–VG parameters for illustration)
model = AlphaVGModel(
    lam=10.0,
    alpha=0.4,
    a_plus=20.4499,
    a_minus=24.4499,
    S1=100.0,
    S2=96.0,
    T=1.0,
    r=0.1,
    q1=0,
    q2=0,
)

pricer = SpreadPricer(model)

# Benchmark across multiple strikes
Ks = np.arange(0.4, 4.1, 0.4, dtype=float)

for K in Ks:
    t0 = time.perf_counter()
    v = pricer.price(K, tol=1e-6, U0=40.0, h_max=0.02, info=False)
    dt = time.perf_counter() - t0
    print(f"{dt:.4f}s used, K={K}: {v}")
```

### Running the Benchmark

```bash
python main.py
```

---

## Extending to New Models

To add a model, implement `SpreadModel.phi`. For example, a bivariate Black–Scholes:

```python
import numpy as np
from .base import SpreadModel

class BivariateBSModel(SpreadModel):
    def __init__(self, vol1, vol2, rho):
        self.v1, self.v2, self.rho = float(vol1), float(vol2), float(rho)

    def phi(self, u1, u2, T, r, q1, q2, convention="rn"):
        v1, v2, rho = self.v1, self.v2, self.rho
        mu1 = (r - q1 - 0.5*v1*v1)*T
        mu2 = (r - q2 - 0.5*v2*v2)*T
        qf  = 0.5*T*(v1*v1*u1*u1 + v2*v2*u2*u2 + 2*rho*v1*v2*u1*u2)
        return np.exp(1j*(u1*mu1 + u2*mu2) - qf)
```

No changes to the pricer are required.

---

## Benchmark Protocol

* **Parameter sweeps** over strikes $K$, maturities $T$, and model parameters.
* **Diagnostics** recorded at termination:

  * $U$, $n$, effective $h=2U/n$, $\omega$, convention,
  * discretization indicator $|v(2n)-v(n)|$,
  * tail indicator $|v(U_{\text{new}})-v(U_{\text{old}})|$,
  * runtime statistics and environment (Python/NumPy/SciPy versions).

These diagnostics enable comparison across models and settings with a common accuracy target.

## Reference

* T. R. Hurd and Z. Zhou, *A Fourier Transform Method for Spread Option Pricing* (preprint/working paper).
  This repository implements their payoff transform and associated 2D inversion framework.


