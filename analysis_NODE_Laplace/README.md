
# 📁 Folder Overview

This folder contains the scripts to run the comparison between the MOD algorithm and the Laplace approximation in the purely data-driven case -the models are pure Neural Ordinary Differential Equations (NODEs)- for the three dynamical systems: **damped oscillator**, **Lorenz system**, and **Lotka-Volterra**. The structure includes configuration files, result analyses, launcher scripts, and utility modules.

---

## 📂 Folder Structure

### 📊 Results Folders

- **`analysis_results_damped/`**
- **`analysis_results_lorenz/`**
- **`analysis_results_lv/`**  
  These folders contain the analysis of the results of the method proposed for each of the respective systems.

- **`results_MOD/`**  
  This folder contains the results as `.jld` files of the MOD algorithm. For each dynamical system, the method was run starting from 10 different initializations, resulting in 10 corresponding result folders.

- **`trainresults_Laplace/`**  
  This folder contains the results of Laplace approximation on the three different dynamical systems. One random model from each standard ensemble is used as the starting point for our method.

---

## ⚙️ Configuration and Utility Scripts

- `configurations_damped.jl`  
- `configurations_lorenz.jl`  
- `configurations_lv.jl`  
  Julia scripts containing parameter settings or simulation options for each system.

- `ConfidenceEllipse.jl`  
  Julia module containing functions to work with confidence ellipses of a multivariate normal distribution.

- `out_of_domain_variability.jl`  
- `out_of_domain_variability_3d.jl`  
  Julia modules for analyzing confidence ellipses of vector field predictions derived from an ensemble of models over a given region of the state space. The first script is customized for 2D vector fields, and the second for 3D vector fields.

- `data_generator/`  
  Scripts used for generating synthetic data for training.

---

## 🚀 Launcher Scripts

- `launcher_laplace_lv.bat`  
- `launcher_laplace_damped.bat`  
- `launcher_laplace_lorenz.bat`  
  Bash scripts to replicate the Laplace approximation for each dynamical system.

---

## 📈 Laplace approximation implementation

- `laplace_lv.jl`  
- `laplace_damped.jl`  
- `laplace_lorenz.jl`  
  Main scripts that implement the Laplace approximation method.

---

## 🧬 Project Metadata

- `Manifest.toml`  
- `Project.toml`  
  Julia environment files specifying project dependencies and versions.

---