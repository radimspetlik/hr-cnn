# HR-CNN
Spetlik, R., Franc, V., Cech, J. and Matas, J. (2018) Visual Heart Rate Estimation with Convolutional Neural Network. In Proceedings of British Machine Vision Conference, 2018

See http://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/ for the original paper and the ECG-Fitness dataset.

This repo is being constructed. You can monitor the progress bellow.

## Progress
- [x] Document the installation.
- [ ] Convert models to a reasonable format.
- [ ] Create public evaluation scripts.
- [ ] Create public learning scripts.

## Installation
We support only the following plug-and-play installation. You don't have to follow the steps bellow, but it may happen that it will not work :( This setup should work regardless of the operation system (i.e. Windows and Linux is OK).
1. Install _miniconda_ http://lmgtfy.com/?q=miniconda+install.
1. Update _miniconda_ with `conda update -n base conda`.
1. Create a _Python 2.7_ environment _hr-cnn_ with `conda create -n hr-cnn python=2.7`.
1. Activate the environment with `source activate hr-cnn`.
1. Install packages.
	1. Install _Pytorch_ http://lmgtfy.com/?q=install+pytorch.
	1. Install _docopt_ with `conda install docopt`.
	1. Install _scipy_ with `conda install scipy`
	1. Install _h5py_ with `conda install h5py`
	1. Install _opencv_ with `conda install opencv`
	1. Install _boost 1.65.1_ with `conda install boost=1.65.1`
1. Clone the repo to a directory of your preference with `git clone git@github.com:radimspetlik/hr-cnn.git`.
1. Add the data.
	1. Download the models from http://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/models.zip and extract them to `data/models/`.
	1. Copy the contents of the `bbox` directory (distributed in the 7zipped ECG Fitness database) to `bob/db/ecg_fitness/data/bbox/`.
1. Run the test script with `python hr-cnn/test.py`. The network will evaluate a short sequence attached in the repo. You should get the following results:

```
[INFO] Root Mean Squared Error = ?
[INFO] Mean of error-rate percentage = ?
[INFO] Mean absolute error = ?
[INFO] Pearson's correlation = ?
[INFO] Pearson's correlation significance = ?
```
