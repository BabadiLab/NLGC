### Steady state Kalman filtering

Instead of recursive Kalman updates

$\Sigma_{n|n-1} = A\Sigma_{n-1|n-1}A^\top + Q$

$K_n = \Sigma_{n|n-1} F^\top(F\Sigma_{n|n-1} F^\top + R)^{-1}$

$\Sigma_{n|n} = \Sigma_{n|n-1}  - K_nF\Sigma_{n|n-1}$

solve the discrete **ARE**

$\Sigma^{-(r)} = A\Sigma^{-(r)}A^\top  - A\Sigma^{-(r)} F^\top(F\Sigma^{-(r)} F^\top + R)^{-1}F\Sigma^{-(r)}A^\top + Q$

to solve for $\Sigma^{-(r)}$ and then compute the following entities:

1. Kalman gain, $K = \Sigma^{-(r)} F^\top(F\Sigma^{-(r)} F^\top + R)^{-1}$

2. steady state error covarince,
$$\begin{aligned}\Sigma^{(r)}&=\Sigma^{-(r)} - KF\Sigma^{-(r)}\\ 
&=\Sigma^{-(r)} - \Sigma^{-(r)} F^\top(F\Sigma^{-(r)} F^\top + R)^{-1}F\Sigma^{-(r)}\end{aligned}$$ and

3. smoothed gain,
$B =\Sigma^{(r)}A^\top{\Sigma^{-(r)}}^{-1}$

## log-likelihood computation
#### model
$y_n = Fx_n + v_n$

$x_n = Ax_{n-1} + w_n$

#### log-likelihood expressions
$p(y^N|\theta) = \int p(x^N, y^N|\theta)dx^N = \int p(y^N|x^N, \theta)p(x^N|\theta)dx^N$is intractable to compute, instead use the following relation: 
$$p(y^N|\theta) = \frac{p(x^N, y^N|\theta)}{p(x^N|y^N, \theta)}$ , $\forall x^N$$

But that requires one candidate sample-path, $x^N$, which we choose to sample from the following distribution, so that the denominator is bound away from zero. 

[1]_ gives following easy decomposition of the distribution, to make the sampling easy.
$$\begin{aligned}p(x^N|y^N, \theta) &= \prod p(x_n|x_{n+1},\cdots,x_N, y^N,\theta)p(x_N|y^N, \theta)\\ &= |\hat{\Sigma}^{(r)}|^{-(N-1)/2}\exp\frac{1}{2}\sum_{1}^{N-1}\left((x_n-\hat{x}_{n|N})^\top{\hat{\Sigma}^{(r)}}^{-1}(x_n-\hat{x}_{n|N})\right)|\Sigma^{(r)}|^{-1/2}\exp\frac{1}{2}\left(-(x_N-x_{N|N})^\top{\Sigma^{(r)}}^{-1}(x_N-x_{N|N})\right)\end{aligned}$$
with 

$\hat{x}_{n|N} =(I-CA)x_{n|n} + Cx_{t+1}=x_{n|n} +C(x_{t+1}-x_{n+1|n})$

$\hat{\Sigma}^{(r)}=\Sigma^{(r)} - CA\Sigma^{(r)}=\Sigma^{(r)} - C{\Sigma^{-(r)}}C^{\top}$ is different from $\Sigma^{(+r)}$

$C = \Sigma^{(r)}A^\top(A\Sigma^{(r)}A^\top + Q)^{-1} = \Sigma^{(r)}A^\top{\Sigma^{-(r)}}^{-1}=B$

Now consider $x^N=\{x_{n|N}\}_{n=1}^{N}$ as a sample path, then in the above equation the summation term, i.e., whatever is inside the $\exp$ term becomes zero. We then just need to evaluate two determinant terms. :) 

$p(x^N, y^N|\theta) \propto |R|^{-N/2}|Q|^{-N/2}\exp\frac{1}{2}\sum_{n=1}^{N}\left(-(y_n-Fx_n)^{\top} R^{-1}(y_n-Fx_n)-(x_n-Ax_{n-1})^\top Q^{-1}(x_n-Ax_{n-1})\right)$


## References:
[1]_ Frühwirth-Schnatter, Sylvia (1992) Data Augmentation and Dynamic Linear Models.URL: https://epub.wu.ac.at/id/eprint/392

