# HR-CNN
Spetlik, R., Franc, V., Cech, J. and Matas, J. (2018) Visual Heart Rate Estimation with Convolutional Neural Network. In Proceedings of British Machine Vision Conference, 2018

See http://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/ for the original paper and the ECG-Fitness dataset.

This repo is being constructed. You can monitor the progress bellow.

## Progress
- [ ] Document the installation.
- [ ] Convert models to a reasonable format.
- [ ] Create public evaluation scripts.
- [ ] Create public learning scripts.

## Installation
We support only the following plug-and-play installation. You don't have to follow the steps bellow, but it may happen that it will not work :( This setup should work regardless of the operation system (i.e. Windows and Linux is OK).
1. Install _miniconda_ http://lmgtfy.com/?q=miniconda+install.
1. Create a _Python 2.7_ environment _hr_cnn_ `conda create -n hr-cnn python=2.7`.
1. Activate the environment `source activate hr-cnn`.
1. Install _Pytorch_ http://lmgtfy.com/?q=install+pytorch.
1. Clone the repo to a directory of your preference `git clone git@github.com:radimspetlik/hr-cnn.git`.
1. Run the test script with `python hr-cnn/test.py`. The network will evaluate a short sequence attached in the repo. You should get the following results:

```
[INFO] Root Mean Squared Error = ?
[INFO] Mean of error-rate percentage = ?
[INFO] Mean absolute error = ?
[INFO] Pearson's correlation = ?
[INFO] Pearson's correlation significance = ?
```
