from cmp.nrppg.db.datasetworkers import Dataset
from cmp.nrppg.cnn.ModelLoader import ModelLoader
from cmp.nrppg.experiments.compute_performance import main as compute_performance
from cmp.nrppg.experiments.frequency_analysis import main as freqeuency_analysis
from cmp.nrppg.experiments.extract_pulse_cnn import main as extract_pulse_main
import logging
import os
import torch


def perform_experiments(database, protocol='all', qf=None):
    __logging_format__ = '[%(levelname)s] %(message)s'
    logging.basicConfig(format=__logging_format__)
    logger = logging.getLogger("cnn-experiments_log")
    logger.setLevel(logging.INFO)

    logger.info('Performing experiments on %s database, protocol %s.' % (database, protocol))

    compressed = ''
    if 'compressed' in database:
        compressed = '-compressed'
        database = database.split('-')[0]

    fps = 0
    metrics = ['whole']
    if database == Dataset.PURE:
        fps = 30
    elif database == Dataset.COHFACE:
        fps = 20
    elif database == Dataset.HCI:
        fps = 61
    elif database == Dataset.ECG_FITNESS:
        fps = 30

    # for which subset to extract signals?
    subset = ''  # '--subset '

    start_end = ''  # ''--start 306 --end 2136'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_folder = os.path.join(current_dir, 'experiments.rppg.base/bin/')
    db_dir = os.path.join(hr_directory, 'db', str(database))

    best_extractor_model = {}
    extractor_model_trained_on_db = 'ecg-fitness'

    best_extractor_model['pure'] = '08-03-2018_11-31-02-752459_arch=FaceHRNet09V2ELURGB_snr_median_best_epoch_min-std-0.10_em-after-1002_batch-size-300'
    best_extractor_model['cohface'] = '08-03-2018_12-22-31-356393_arch=FaceHRNet09V2ELURGB_snr_median_best_epoch_min-std-0.10_em-after-1002_batch-size-300'
    best_extractor_model['hci'] = '08-03-2018_19-21-12-541339_snr_median_best_epoch_architecture-FaceHRNet09V2ELURGB_min-std-0.10_em-after-1002_batch-size-500'
    best_extractor_model['ecg-fitness'] = '08-05-2018_23-04-07-384974_arch=FaceHRNet09V4ELURGB_lr=1E-05_batch-size=300_fine_tuning_extractor_val_mae_best'

    best_estimator_model = {}
    estimator_model_trained_on_db = database + compressed
    best_estimator_model['pure'] = '25-04-2018_19-06-03-277330_arch=SNREstimatorNetMonteCarlo_lr=1E-01_batch-size=600_db-name=pure_epoch=386_val_mae_best'
    best_estimator_model['pure-compressed'] = '22-04-2018_19-26-38-212160_arch=SNREstimatorNetMonteCarlo_lr=1E-01_batch-size=600_db-name=pure_epoch=371_val_mae_best'
    best_estimator_model['cohface'] = '17-04-2018_19-50-38-274449_arch=SNREstimatorNetMonteCarlo_lr=1E-01_batch-size=600-cohface_epoch=362_trn-val_mae_best'
    best_estimator_model['hci'] = '15-04-2018_15-27-56-539166_arch=SNREstimatorNetMonteCarlo_lr=1E-01_batch-size=600-hci_epoch=273_val_rmse_avg_best'
    best_estimator_model['ecg-fitness'] = '09-05-2018_09-52-57-210538_arch=SNREstimatorNetMonteCarlo_lr=1E-02_batch-size=300_fine-tuning_estimator_val_mae_best'

    batch_size = 600
    extractor_model_name = best_extractor_model[extractor_model_trained_on_db]
    estimator_model_name = best_estimator_model[estimator_model_trained_on_db]

    extractor_model_path = os.path.join(models_dir, extractor_model_name)
    estimator_model_path = os.path.join(models_dir, estimator_model_name)

    base_expe_dir = os.path.join(hr_directory, 'experiments', 'cnn', str(database) + compressed + '-' + str(protocol)
                    + '_evaluated_by_%s_cnn' % (extractor_model_trained_on_db), '_'.join(extractor_model_name.split('_')[:5]))
    faces_dir = os.path.join(hr_directory, 'experiments', 'cnn', str(database) + compressed + '-face-192x128')

    pulse_dir = os.path.join(base_expe_dir, 'pulse')
    hr_dir = os.path.join(base_expe_dir, 'hr')
    stats_dir = os.path.join(base_expe_dir, 'stats')
    results_dir = os.path.join(base_expe_dir, 'results')

    # fixed parameter
    n_segments = 16
    nfft = 8192

    id = 0
    pool_size = 1

    extractor_model, rgb = ModelLoader.load_model(extractor_model_path, 'extractor', True)
    extractor_model.cuda()
    extractor_model.eval()
    # signals extraction
    extract_pulse_main(extractor_model, rgb, id, pool_size,
                       str(database) + ' --overwrite -v --plot --protocol ' + str(protocol) + ' ' + str(start_end) + ' ' + str(
                           subset) + ' --cnn-model-path ' + str(extractor_model_path) + ' --dbdir ' + str(
                           db_dir) + ' --faces-dir ' + str(
                           faces_dir) + ' --outdir ' + str(pulse_dir) + ' --batch-size ' + str(batch_size))

    estimator_architecture_name = estimator_model_name.split('_')[2].split('=')[1]
    estimator_model, rgb = ModelLoader.initialize_model(estimator_architecture_name, 'estimator', True)
    if "MonteCarlo" in estimator_architecture_name:
        mc_conf = torch.load(os.path.join('_'.join(estimator_model_path.split('_')[:7]) + '_monte-carlo-configuration'))
        try:
            estimator_model.setup(mc_conf['active_layers'], mc_conf['max_pool_kernel_size'], mc_conf['conv_kernel_size'],
                                  mc_conf['conv_filter_size'])
        except AttributeError as e:
            estimator_model.module.setup(mc_conf['active_layers'], mc_conf['max_pool_kernel_size'], mc_conf['conv_kernel_size'],
                                         mc_conf['conv_filter_size'])

    estimator_model = ModelLoader.load_parameters_into_model(estimator_model, estimator_model_path, True)
    estimator_model.cuda()
    estimator_model.eval()

    # computing heart-rate
    freqeuency_analysis(estimator_model, id, pool_size, metrics, hr_directory, fps, str(database)
                             + ' --overwrite --plot --dbdir '
                             + str(db_dir) + ' --protocol ' + str(protocol) + ' ' + str(subset) + ' --indir ' + pulse_dir
                             + ' --outdir ' + hr_dir + ' --framerate ' + str(fps) + ' --nsegments ' + str(n_segments)
                             + ' --nfft ' + str(nfft) + ' --stats-outdir ' + str(stats_dir) + ' -v')

    logger.info('Frequency analysis done.')

    compute_performance(qf, metrics, str(database) + ' --overwrite --plot --dbdir ' + str(db_dir) + ' -v --protocol ' + str(protocol)
                            + ' --subset train --indir ' + hr_dir + ' --outdir ' + results_dir)
    compute_performance(qf, metrics, str(database) + ' --overwrite --plot --dbdir ' + str(db_dir) + ' -v --protocol ' + str(protocol)
                            + ' --subset test --indir ' + hr_dir + ' --outdir ' + results_dir)


if __name__ == "__main__":
    hr_directory = os.path.join('~', 'hr-cnn', 'data')
    models_dir = os.path.join(hr_directory, 'models')

    dataset = Dataset.ECG_FITNESS

    perform_experiments(dataset, 'test')