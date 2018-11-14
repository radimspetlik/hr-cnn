# HR-CNN
Spetlik, R., Franc, V., Cech, J. and Matas, J. (2018) Visual Heart Rate Estimation with Convolutional Neural Network. In Proceedings of British Machine Vision Conference, 2018

See http://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/ for the original paper and the ECG-Fitness dataset.

This repo is being constructed. You can monitor the progress bellow.

## Progress
- [x] Document the installation.
- [x] Convert models to a reasonable format.
- [x] Create public evaluation scripts.
- [ ] Create public learning scripts.

## Installation
We support only the following plug-and-play installation. You don't have to follow the steps bellow, but it may happen that it will not work :( This setup should work regardless of the operation system (i.e. Windows and Linux is OK).
1. Clone the repo to a directory of your preference with `git clone git@github.com:radimspetlik/hr-cnn.git`.
1. Change directory to the clonned repo.
1. Install _miniconda_ http://lmgtfy.com/?q=miniconda+install.
1. Run `conda env update --file environment.yml` *OR* manually:
	1. Create a _Python 2.7_ environment _hr-cnn_ with `conda create -n hr-cnn python=2.7`.
	1. Install _docopt_ with `conda install docopt`.
	1. Install _scipy_ with `conda install scipy`
	1. Install _h5py_ with `conda install h5py`
	1. Install _opencv_ with `conda install opencv`
	1. Install _boost 1.65.1_ with `conda install boost=1.65.1`
	1. Install _bob.blitz_ with `conda install bob.blitz`
	1. Install _lmdb-python_ `conda install lmdb-python`
1. Activate the environment with `source activate hr-cnn`.
1. Install _Pytorch_ http://lmgtfy.com/?q=install+pytorch.	
1. Add the data.
	1. Download the models from http://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/models.zip and extract them to `data/models/`.
	1. Copy the contents of the `bbox` directory (distributed in the 7zipped ECG Fitness database or available at https://goo.gl/aXDQiy) to `bob/db/ecg_fitness/data/bbox/`.
	1. Copy the contents of the `test_h5_faces.zip` (available at https://goo.gl/9iw3LY) to `data/experiments/cnn/ecg-fitness-face-192x128/15/01/`.
	1. Copy the contents of the `ecg-fitness_lmdb.zip` (availabe at https://goo.gl/MFLXH2) to `data/db`.
1. Run the _evaluation_ test script with `python test-evaluation.py`. The network will evaluate two sequences attached in the repo. You should get the following results:
```
[INFO]==================
[INFO]=== STATISTICS-whole train ===
[INFO]Root Mean Squared Error-whole = 8.38
[INFO]Mean of error-rate percentage-whole = 0.09
[INFO]Mean absolute error-whole = 8.30
[INFO]Pearson's correlation-whole = nan
[INFO]Pearson's correlation-whole significance = nan
[INFO]==================
[INFO]=== STATISTICS-whole test ===
[INFO]Root Mean Squared Error-whole = 8.38
[INFO]Mean of error-rate percentage-whole = 0.09
[INFO]Mean absolute error-whole = 8.30
[INFO]Pearson's correlation-whole = nan
[INFO]Pearson's correlation-whole significance = nan
```
*WARNING* - the computations are very GPU memory-demanding. Running the test script requires at least 12GB of GPU memory. If you don't have enough memory, try changing the `batch_size` variable in the `test.py` script.	
1. Run the _extractor training_ test script with `python -u test-extractor-training.py --plot-after 10 --batch-size 300 --epochs 2 --lr 0.0001 --x-lmdb-path-train 'data/db/ecg-fitness_face_linear-192x128_batch-300_test-train_X_lmdb' --y-lmdb-path-train 'data/db/ecg-fitness_face_linear-192x128_batch-300_test-train_y_lmdb' --x-lmdb-path-validation 'data/db/ecg-fitness_face_linear-192x128_batch-300_test-train_X_lmdb' --y-lmdb-path-validation 'data/db/ecg-fitness_face_linear-192x128_batch-300_test-train_y_lmdb' --output-to 'data/models/' --plots-path 'data/plots/' --net-architecture 'FaceHRNet09V4ELURGB'`. You should get the follwing results:
```
[INFO][0000][TRN] 5.294248 MAE: 39.3 MSE: 2127.2, 7056.0 (6s)
[INFO][0000][VAL] 5.288296 MAE: 27.8 MSE: 1185.5, 3249.0
[INFO][0001][TRN] 5.290246 MAE: 28.2 MSE: 1237.2, 4624.0 (6s)
[INFO][0001][VAL] 5.285192 MAE: 30.4 MSE: 1322.6, 3249.0
[INFO][0002][TRN] 5.283029 MAE: 24.2 MSE: 1131.5, 3844.0 (6s)
[INFO][0002][VAL] 5.274898 MAE: 17.2 MSE: 781.6, 2601.0
[INFO]Succesfully finished...
```
*WARNING* - the scripts assume that the batch size corresponds to a sample size with which the LMDB dataset was created - in our case, this is 300 frames per sample.
*WARNING* - note that in the BMVC paper, a SNR in equation (3) is presented as the extractor learning objective function. The first extractor network was learned with exactly this criterion. However, in the latest scripts presented in this repo (which were used in my diploma thesis available at https://dspace.cvut.cz/bitstream/handle/10467/77090/F3-DP-2018-Spetlik-Radim-robust_visual_heart_rate_estimation.pdf) a cross-entropy objective function is used.

## bob.rppg.base

My scripts are using a minimalist hackish version of https://pypi.org/project/bob.rppg.base/. I am sorry for that. Be sure to checkout their repo. Just to be absolutely sure - everything you need from their repo to run my scripts is included in my repo.