# NLGC

Understanding how the brain processes the sensory information requires the ability to draw causal connections across different cortical areas of the brain. In the study, we propose a statistical framework, called network localized Granger causality (NLGC), to capture causal relationships at the cortical level from magnetoencephalography (MEG) data. As opposed to the conventional methods, so-called the two-stage procedure which in fact source activities need to be estimated first followed by the connectivity analysis given the source estimates, we directly capture the causal links without an intermediate source localization step. Simulation results show that the NLGC outperforms the two-stage procedure both in terms of hit-rate and detecting less spurious interactions. Furthermore, NLGC is sufficiently robust against model mismatches, network size, and low signal-to-noise ratio. Fig. 1 shows an overview of the NLGC pipeline. For more details, check [1] and [2].

|<img src="https://user-images.githubusercontent.com/95252372/153320252-fe1d1c7f-e882-4c82-a7bb-807b5473840c.jpg" width="70%" alt=""> | 
|:--:| 
| Fig 1. NLGC pipeline. In contrast to conventional two-stage procedures, NLGC directly captures the causal interactions without an intermediate source localization step. |

This repository includes the implementation of our proposed direct causal inference algorithm in python (version 3.8 and above).

# Requirements

Eelbrain ([instructions and installation](https://github.com/christianbrodbeck/Eelbrain/wiki/Installing#release))

# Installation

After successfully installing eelbrain, one can install NLGC package using 'pip' (or any other convinient methods).
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
.
.
# reading the data and pre-processing...
.
.
nlgc_obj = nlgc.nlgc_map(subject, evoked, forward, noise_cov, src_target, order=order, n_eigenmodes=n_eigenmodes, patch_idx=patch_idx, n_segments=n_segments, lambda_range=lambda_range, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor, var_thr=var_thr, cv=5)
```
This returns an object which includes connectivity matrix, estimated parameters, and some related variables. For example, the connectivity matrix (J-values) is given by
```
J = nlgc_obj.get_J_statistics()
```
For more details, check the description of 'nlgc_map()'.

*Note(1): argumetns of 'nlgc_map()' such as forward model and evoked follow the standard [MNE-experiment](https://eelbrain.readthedocs.io/en/stable/experiment.html) pipeline.

*Note (2): to plot the connectivity map over a brain, one can use [connectome plot](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_connectome.html) (e.g. Fig. 2 panel A, or Fig. 3).

# Results

Fig. 2 illustrates the comparison of NLGC with the two-stage procedures using three well-known source localization techniques: [MNE](https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html), [dSPM](https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html), and [Champagne](https://mne.tools/stable/generated/mne.inverse_sparse.gamma_map.html#mne.inverse_sparse.gamma_map). Panel A shows the captured causal network for one configurations. In Panel B, the ROC curves (hit-rate vs. false alarm) is depicted for NLGC and the other three methods in different scenarios: 1) exact vs. relaxed localization: in the relaxed version, if the link A'->B' is detected and the true link was A->B, we recognize it as a hit if A' and B' belong to the neighboring sources of A and B, respectively. This way, we neglect some errors happend due to the mis-localization. 2) model mis-match vs. no model mis-match: in the no mis-math case, the forward model used to generate the data is the exact same fed to the algorithm. On the other hand, in model mis-match case (which is a typical practical scenario), the forward model utilized in the estimation procedures is a deviated version of the true forward model. Finally, Panel C compares the perfotmace of these frameworks with respect to the signal-to-noise ratio (SNR). Overall, this simulation suggests NLGC can be reliably used instead of the conventional two-stage procedures as it is robust with respect to model mis-match and SNR, as well as network size. For more details, please check [1]. 
|<img src="https://user-images.githubusercontent.com/95252372/153320153-8066de05-66b8-4395-9e65-fdae041c98f7.jpg" width="90%" alt=""> | 
|:--:| 
| Fig 2. Comparison of NLGC with the two-stage procedures. |

Fig. 3 demonstrates the application of NLGC on experimentally recorded data from a tone listening vs. resting state task over a group of younger and older participants. In this figure, frontal -> temporal (red) and temporal -> frontal (green) connections are plotted at the group- (left panel) and individual-level in 0.1-8 Hz frequency band. According to the figure, in tone processing, for both age groups, frontal -> temporal (top-down) interactions have more contribution in the established netwrok while in the resting state, more temoporal -> frontal (bottom-up) connections appear. More details are available in [1].

|<img src="https://user-images.githubusercontent.com/95252372/153009581-ff7962b7-897e-458a-b611-edd8c092cc0c.jpg" width="90%" alt=""> | 
|:--:| 
| Fig 3. Delta and Theta band analysis of a tone processing vs. resting-state task for younger and older participants. |

# Contact

We keep developing the package over time. Feel free to ask any other functionality if it is not already implemented and/or report if anything is broken. You can contact the authors under these email addresses: behrad@umd.edu (Behrad Soleimani), pdas6@mgh.harvard.edu (Proloy Das).


# Term of Use

This python package is a free software under BSD 3-Clause License. Whenever you use this software to produce a publication or talk, please cite the following references.


# References

[1] B. Soleimani, P. Das, I.M. D. Karunathilake, S. E. Kuchinsky, J. Z. Simona and B. Babadi, "[NLGC: Network Localized Granger Causality with Application to MEG Directional Functional Connectivity Analysis](https://www.biorxiv.org/)", ***.

[2] B. Soleimani, P. Das, J. Kulasingham, J. Z. Simon and B. Babadi, "[Granger Causal Inference from Indirect Low-Dimensional Measurements with Application to MEG Functional Connectivity Analysis](https://ieeexplore.ieee.org/abstract/document/9086218)", 2020 54th Annual Conference on Information Sciences and Systems (CISS), 2020, pp. 1-5, doi: 10.1109/CISS48834.2020.1570617418.

