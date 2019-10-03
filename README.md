# HPC Automated Pipeline Processing Yotta Neurons

This library aim to provide integration of different libraries used for neuroscience with the current US HPC facilities.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

* Neuroglancer
* CloudVolume
* Knossos Utils
* ffn
* etc

### Building your preamble at Cooley HPC.

A step by step series of examples that tell you how to get a development env running

Say what the step will be


### Building your preamble at Theta HPC.

A step by step series of examples that tell you how to get a development env running


## List of Macros in workflow form

### Data Acquisition
- [x] Generate dataset Preview
- [ ] Generate dataset quality check
- [ ] Generate bad-image list

### Tile Image Processing
- [x] Correct each image individually
- [x] Correct brightness individually

### Slice Montage (X-Y dimension)
- [x] Pairwise aligment
- [x] Global Optimzation
- [x] Stitching

### Alignment (Z dimension)
- [x] Pairwise aligment
- [x] Global Optimzation
- [x] Cube rendering

### Visualization 
- [x] Cube cropping (removing borders)
- [x] Cube neuroglancing
- [x] Inset cropping

### Training
- [x] Inset Manual annotation
- [x] FFN compute_partitions
- [x] FFN build_coordinates
- [x] FFN train

### Inference
- [x] Splice data
- [x] Inference
- [x] Merge

### Final Visualization
- [x] Precompute data
- [x] Precompute Labels
- [x] Generate Mesh

### Other Macros
- [x] Knossos injection/extraction
- [x] webKnossos injection/extraction



## Deployment

The folder macros will have terminal callable scripts and the folder autoem will have the libraries involved.
On a crude fashion, each item of the list bellow should be a blackbox (script with defined inputs/outputs).
Those scripts should be either callable from a MPI compatible machine or a single workstation.


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Rafael Vescovi** - *Initial work*
* **Hanyu Li** - *Initial work*
* **Murat Keceli** - *Initial work*
* **Bobby Kasthuri** - *Initial work*
* **Nicola Ferrier** - *Initial work*
* **Rafael Vescovi** - *Initial work*

See also the list of [contributors](https://github.com/ravescovi/autoem/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* University of Chicago
* Argonne National Laboratories
* Harvard University
* Google
* Princeton University
