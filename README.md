# STDWeb - Simple Transient Detection for the Web

This is a simple web-based tool for a quick-look photometry and transient detection in astronomical images. It is based on [STDPipe](https://github.com/karpov-sv/stdpipe) library and tries to implement a self-consistent and mostly automatic data analysis workflow from its routines.

It currently allows you to:

- Upload your own FITS files, or analyze some files already on the server
- Do basic pre-processing and masking
- Detect objects in the image and do astrometric calibration, either by blind mathcing or refining existing solution
- Photometrically calibrate the image using one of supported reference catalogues
- Subtract either user-provided or automatically downloaded template images
- Do forced photometry for a specified target in either original or difference image
- Do (experimental) transient detection in difference image

If you want to better understand the routines used for it, please consult [STDPipe documentation](https://stdpipe.readthedocs.io/) and [example notebooks](https://github.com/karpov-sv/stdpipe/tree/master/notebooks)

