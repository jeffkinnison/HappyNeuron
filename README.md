# autoem

Automatic Electron Microscopy or AutoEM for short is a set of macros to create a full EM pipeline.

The folder macros will have terminal callable scripts and the folder autoem will have the libraries involved.

On a crude fashion, each item of the list bellow should be a blackbox (script with defined inputs/outputs).

Those scripts should be either callable from a MPI compatible machine or a single workstation.

##List of Macros in workflow form

###Data Acquisition
- [x] 1 - Acquire full dataset
- [ ] 1.1 (SEM) - Generate dataset Preview
- [ ] 1.2 (SEM) - Generate dataset quality check
- [ ] 1.3 (SEM) - Generate bad-image list
- [x] 1.4 (SEM) - Reaquire data

###Processing
- [ ] 2.1 (ALCF) - Correct each image individually
- [ ] 2.2 (ALCF) - Correct brightness individually

###Aligment
- [ ] 3.1 (ALCF) - Pairwise aligment
- [ ] 3.2 (ALCF) - Global Optimzation
- [ ] 3.3 (ALCF) - Cube rendering (applying aligment)
- [ ] 3.4 (ALCF) - Cube cropping
- [ ] 3.5 (ALCF) - Cube neuroglancing
- [ ] 3.6 (ALCF) - Inset cropping

###Training
- [ ] 4.0 (Local) - Inset Manual annotation
- [ ] 4.1 (ALCF) - FFN compute_partitions
- [ ] 4.2 (ALCF) - FFN build_coordinates
- [ ] 4.3 (DGX) - FFN train

###Inference
- [ ] 5.0 (ALCF) - Splice data
- [ ] 5.1 (ALCF) - Inference
- [ ] 5.2 (ALCF) - Merge

###Visualize
- [ ] 6.0 (ALCF) - Precompute data
- [ ] 6.1 (ALCF) - Precompute Labels
