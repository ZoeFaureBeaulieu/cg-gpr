# Research data supporting "Coarse-grained versus fully atomistic machine learning for zeolitic imidazolate frameworks"

[![DOI](https://zenodo.org/badge/637702153.svg)](https://zenodo.org/badge/latestdoi/637702153)

This repository supports the manuscript:

<div align="center">

> **[Coarse-grained versus fully atomistic machine learning for zeolitic imidazolate frameworks](https://arxiv.org/abs/2305.05536)**\
> _[Zoé Faure Beaulieu](https://twitter.com/ZFaureBeaulieu), [Thomas Nicholas](https://twitter.com/thomascnicholas), [John Gardner](https://twitter.com/jla_gardner), [Andrew Goodwin](https://goodwingroupox.uk/), and [Volker Deringer](http://deringer.chem.ox.ac.uk)_

</div>

See the [sister repository](https://github.com/tcnicholas/hZIF-data) for 
details of the database used to obtain the results in the [mansucript](https://arxiv.org/abs/2305.05536).

## License <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

---
## Repository Overview

* **[hZIF-data](hZIF-data)** is a git submodule pointing to Thomas Nicholas' [hypothetical ZIF dataset](https://github.com/tcnicholas/hZIF-data).
* **[scripts](scripts)** contains the Python scripts required to run all the experiments.
* **[notebooks](notebooks)** contains the notebooks used to generate the plots in the paper.
* **[results](results)** contains all the raw data needed to recreate the results from the paper.

---

## Reproducing our results

### **1. Clone the repository**
```bash
git clone --recurse-submodules git@github.com:ZoeFaureBeaulieu/cg-gpr.git
cd cg-mofs
```

### **2. Install dependencies**
All the dependencies (and their versions) used can be found in [requirements.txt](requirements.txt). To use, first create/activate your virtual environment using conda:
```bash
conda create -n gpr python=3.8 -y
conda activate gpr
```
Then install dependencies using:
```bash
pip install -r requirements.txt
```

### **3. Run an experiment**
At this stage, you should be ready to run any of the scripts and/or notebooks. 

An easy test to check that the code is working is to run a gpr experiment with a very low number of training enviornments:
```bash
python scripts/gpr.py --struct_type cg --hypers_type cg --numb_train 10
```
---
