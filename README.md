# -The-pseudo-spectral-approach-in-Chaosoy
 comparison of the two approaches for the expectation and variance over K  by computing the relative error. 
## Model Problem: Linear Damped Oscillator

Consider the model problem, the linear damped oscillator:

$$
\begin{cases}
\frac{d^2 y}{dt^2}(t) + c \frac{dy}{dt}(t) + ky(t) = f \cos(\omega t) \\\\
y(0) = y_0 \\\\
\frac{dy}{dt}(0) = y_1
\end{cases} \tag{10}
$$

Considering $t \in [0, 10]$, $\Delta t = 0.01$, $c = 0.5$, $k = 2.0$, $f = 0.5$, $y_0 = 0.5$, $y_1 = 0.0$, use the `odeint` function from `scipy.integrate` to discretize the model from Eq. (10).  
(You can also reuse code from previous worksheets or the template from Moodle as starting point.)  
For the upcoming experiments, the output that we are interested in is again $y_0(10)$.  
Assume that $\omega \sim \mathcal{U}(0.95, 1.05)$.

Write a **python** program to propagate the uncertainty in $\omega$ through the model in Eq. (10) using the gPC expansion method from Eq. (5).  
Compute the expansion coefficients from Eq. (6) using the pseudo-spectral approach.

Here, use two approaches:

1. Write your own implementation of the pseudo-spectral approach by implementing Eq. (6)
2. Use the functionalities offered by **chaospy** to compute the coefficients (which uses Eq. (6) under the hood).

---

Consider $K = [1, 2, 3, 4, 5, 6]$ and $N = [1, 2, 3, 4, 5, 6]$ (and even higher if you have the resources); the correspondence is one-to-one.  
Based on these coefficients, compute the expectation and the variance of $y_0(10)$.

Compare to a Monte Carlo reference solution computed with $N_{\text{ref}} = 1\,000\,000$ samples, which is given as:

$$
\athbb{E}_{\text{ref}}[y_0(10)] = [-0.43893703]^T \\\\
\text{Var}_{\text{ref}}[y_0(10)] = [0.00019678]^T
$$

by computing the relative error.

