# NLGC

Understanding how the brain processes the sensory information requires the ability to draw causal connections across different cortical areas of the brain. In the study, we propose a statistical framework, called network localized Granger causality (NLGC), to capture causal relationships at the cortical level from magnetoencephalography (MEG) data. As opposed to the conventional methods, so-called the two-stage procedure which in fact source activities need to be estimated first followed by the connectivity analysis given the source estimates, we directly capture the causal links without an intermediate source localization step. Simulation results show that the NLGC outperforms the two-stage procedure both in terms of hit-rate and detecting less spurious interactions. Furthermore, NLGC is sufficiently robust against model mismatches, network size, and low signal-to-noise ratio. Fig. 1 shows an overview of the NLGC pipeline. For more details, check [1] and [2].

|<img src="https://user-images.githubusercontent.com/95252372/150391276-77ccd762-71b7-48c5-9a34-0a5a77110e44.jpg" width="70%" alt=""> | 
|:--:| 
| Fig 1. NLGC pipeline. In contrast to conventional two-stage procedures, NLGC directly captures the causal interactions without an intermediate source localization step. |

This repository includes the implementation of our proposed direct causal inference algorithm in python (version 3.8 and above).

# Requirements

Eelbrain ([instructions and installation](https://github.com/christianbrodbeck/Eelbrain/wiki/Installing#release))

# Installation

After successfully installing eelbrain, one can install NLGC using 'pip' (or any other convinient methods).
```
pip install NLGC
```
Also, the development version from GitHub can be cloned and installed as follows
```
git clone https://github.com/BabadiLab/NLGC.git
cd NLGC
pip install -e .
```
# How to use

After importing the package, one need to call nlgc_map function with proper arguments as follows
```
import nlgc

# reading the data and pre-processing...

nlgc_obj = nlgc.nlgc_map(subject, evoked, forward, noise_cov, src_target, order=order, n_eigenmodes=n_eigenmodes, patch_idx=patch_idx, n_segments=n_segments, lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor, var_thr=var_thr, cv=5)
```
This returns an object which includes connectivity matrix, estimated parameters, and some related variables. For example, the connectivity matrix (J-values) is given by
```
J = nlgc_obj.get_J_statistics()
```
