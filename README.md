# A novel approach to quantify out-of-distribution uncertainty in Neural and Universal Differential Equations
Code for the paper [S. Giampiccolo, G. Iacca & L. Marchetti. A novel approach to quantify out-of-distribution uncertainty in Neural and Universal Differential Equations](https://doi.org/10.1038/s41540-024-00460-3)

This repository contains the scripts to replicate the analysis of uncertainty quantification performance across various experimental setups described in the main text and in the supplementary material.

---

## 📁 Subfolder Descriptions

### 🔹 `analysis_standard_ensembles/`
This folder contains code to replicate the **motivation case study**:  
- Analysis of 0.95 **prediction interval coverage** for standard ensembles in the **purely data-driven** scenario.
- Applied to all three test numerical test cases:
  - Lotka-Volterra
  - Damped Oscillator
  - Lorenz System

---

### 🔹 `analysis_NODE_maximized/`
This folder contains code to reproduce results of the **proposed MOD (Maximized Out-of-Distribution) ensemble method** in the **purely data-driven** setting (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles vs. standard ensembles on all three test systems.

---

### 🔹 `analysis_Laplace_maximized/`
This folder contains code to reproduce results of the **Laplace approximation method** in the **purely data-driven** setting (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles vs. Laplace approximation on all three test systems.

---

### 🔹 `analysis_NODE_maximized_Lorenz_1500_epochs/`
This folder contains code to reproduce results of the **proposed MOD (Maximized Out-of-Distribution) ensemble method** in the **purely data-driven** setting with 1500 training epochs on the Lorenz system test case (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles trained for 800 epochs vs.  MOD ensembles trained for 1500 epochs.

---

### 🔹 `analysis_NODE_maximized_multiple_training_set/`
This folder contains code to reproduce results of the **proposed MOD (Maximized Out-of-Distribution) ensemble method** in the **purely data-driven** setting, when multiple training sets (each composed by three trajectories) are used (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles vs. standard ensembles on all three test systems.

---

### 🔹 `analysis_NODE_maximized_thre_e-4/`
Same as `analysis_NODE_maximized`, but with a stricter accuracy threshold:  it is used a lower accuracy loss threshold of $\mathcal{L}_{\text{acc}} = 10^{-4}$ to evaluate results under tighter convergence.

---

### 🔹 `analysis_UDE_maximized/`
This folder contains analysis code for the **partially data-driven scenario**, where both **mechanistic parameters** and **neural network parameters** are estimated jointly (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles vs. standard ensembles on all three test systems. We refer to the manuscript for details about which parts of the systems are assumed to be known.

---

### 🔹 `analysis_UDE_with_fixed_mech_par_maximized/`
This folder contains analysis code for the **partially data-driven scenario**, where **mechanistic parameters** are assumed to be known and fixed to their ground truth values (numerical test cases). It compares 0.95 coverage proportion of MOD ensembles vs. standard ensembles on all three test systems. We refer to the manuscript for details about which parts of the systems are assumed to be known.

---

### 🔹 `comparison_among_different_levels_of_prior_knowledge/`
This folder contains analysis code for comparing the coverage proportions on OOD trajectories and state space points on the numerical test cases obtained with different levels of prior knowledge.

---

### 🔹 `biological_test_case/`
This folder contains analysis code for the Computational Biology test case (cell apoptosis model).

---

## 📜 Metadata

- `Manifest.toml` and `Project.toml`  
  Define the Julia environment, dependencies, and package versions for reproducibility.

---

## 📝 Notes

Each folder is self-contained and provides tools to reproduce specific aspects of the experimental evaluation. Refer to the individual README files inside each subfolder for detailed instructions on how to execute the scripts.
