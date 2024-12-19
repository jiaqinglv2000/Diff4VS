# Diff4VS: HIV-inhibiting Molecules Generation with Classifier Guidance Diffusion for Virtual Screening

Thanks [DiGress](https://github.com/cvignac/DiGress) which inspires our code.

## Environment installation

This code was tested with PyTorch 2.0.1, cuda 11.8.

- Download anaconda/miniconda if needed
- Create a rdkit environment that directly contains rdkit:

  ``conda create -c conda-forge -n diff4vs rdkit=2023.03.2 python=3.9``
- `conda activate diff4vs`
- Check that this line does not return an error:

  ``python3 -c 'from rdkit import Chem'``
- Install graph-tool (https://graph-tool.skewed.de/):

  ``conda install -c conda-forge graph-tool=2.45``
- Check that this line does not return an error:

  ``python3 -c 'import graph_tool as gt' ``
- Install the nvcc drivers for your cuda version. For example:

  ``conda install -c "nvidia/label/cuda-11.8.0" cuda``
- Install a corresponding version of pytorch, for example:

  ``pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118``
- Install other packages using the requirement file:

  ``pip install -r requirements.txt``
- Run:

  ``pip install -e .``
- Navigate to the ./src/analysis/orca directory and compile orca.cpp:

  ``g++ -O2 -std=c++11 -o orca orca.cpp``

Note: Code may not work on MacOS.


## Guidance

Train a regressor using at /Diff4VS/src folder`python3 guidance/train_HIV_regressor_CE.py +experiment=regressor_model.yaml dataset=HIV`

Evaluate the guidance model at /Diff4VS/src folder: `python3 main_guidance_HIV_CE.py +experiment=guidance`
