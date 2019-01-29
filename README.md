# autoem

Automatic Electron Microscopy or AutoEM for short is a set of macros to create a full EM pipeline.

The folder macros will have terminal callable scripts and the folder autoem will have the libraries involved.

On a crude fashion, each item of the list bellow should be a blackbox (script with defined inputs/outputs).

Those scripts should be either callable from a MPI compatible machine or a single workstation.

## List of Macros in workflow form

### Data Acquisition
- [ ] Generate dataset Preview
- [ ] Generate dataset quality check
- [ ] Generate bad-image list

### Tile Image Processing
- [ ] Correct each image individually
- [ ] Correct brightness individually

### Slice Montage (X-Y dimension)
- [ ] Pairwise aligment
- [ ] Global Optimzation
- [ ] Stitching

### Alignment (Z dimension)
- [ ] Pairwise aligment
- [ ] Global Optimzation
- [ ] Cube rendering

### Visualization 
- [x] Cube cropping (removing borders)
- [x] Cube neuroglancing
- [x] Inset cropping

### Training
- [ ] Inset Manual annotation
- [ ] FFN compute_partitions
- [ ] FFN build_coordinates
- [ ] FFN train

### Inference
- [ ] Splice data
- [ ] Inference
- [ ] Merge

### Final Visualization
- [ ] Precompute data
- [ ] Precompute Labels
- [ ] Generate Mesh

### Other Macros
- [ ] webKnossos injection/extraction

