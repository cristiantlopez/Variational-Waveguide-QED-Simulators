This folder contains the following files:

1) xxz_global.py performs the adiabatically-assisted VQE of the XXZ model using a global wQED-XX ansatz. The optimization is carried out using PyTorch.
   It calls the files QuantumRegister.py, Operations.py, and QuantumCircuit.py
2) job_xxz_global.py performs the adiabatically-assisted VQE of the XXZ model using a trotterized global wQED-XX ansatz. 
   The optimization is carried out using JAX and Pennylane.
3) job_xxz_all_to_all.py performs the adiabatically-assisted VQE of the XXZ model using an all-to-all wQED-XX ansatz. 
   The optimization is carried out using JAX and Pennylane.
4) job_xxz_hva.py performs the adiabatically-assisted VQE of the XXZ model using an HVA ansatz. 
   The optimization is carried out using JAX and Pennylane.
