# HPC Automated Pipeline Processing Yotta Neurons

This library aim to provide integration of different libraries used for neuroscience with HPC facilities.

Currently the library is organized in 3 levels:
* Unified install and env system for all third part applications used
* Stand alone application wrappers that can be invoked by terminal or imported as a library
* Automated execution functions using ALCF Balsam


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




## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Rafael Vescovi** - *Initial work*
* **Hanyu Li** - *Initial work*
* **Murat Keceli** - *Initial work*
* **Jeff Kinnisson** - *Initial work*
* **Bobby Kasthuri** - *Initial work*
* **Nicola Ferrier** - *Initial work*
* **Rafael Vescovi** - *Initial work*

See also the list of [contributors](https://github.com/ravescovi/autoem/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Argonne National Laboratories
* University of Chicago
* Harvard University
* Princeton University
* Google
