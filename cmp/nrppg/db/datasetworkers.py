import json
import numpy as np
import cv2
import os
import logging
import random
import copy

__logging_format__ = '[%(levelname)s]%(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("results_collector_log")
logger.setLevel(logging.INFO)


class Dataset():
    PURE = 'pure'
    PURE_COMPRESSED = 'pure-compressed'
    ECG_FITNESS = 'ecg-fitness'
    HCI = 'hci'
    COHFACE = 'cohface'
    DB_NAMES = [PURE, PURE_COMPRESSED, ECG_FITNESS, HCI, COHFACE]
    DB_DICT = {PURE: '\\pur{}', PURE_COMPRESSED: '\\makecell{\\pur{}\\\\ {\\scriptsize MPEG-4 Visual}}', ECG_FITNESS: "\\ecf{}", HCI: "\\hci{}", COHFACE: "\\coh{}"}


class ProtocolGenerator(object):
    @staticmethod
    def generate_protocols():
        lighting = {}
        lighting['no-led'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
        lighting['all'] = lighting['no-led']
        lighting['led'] = [11, 12, 13, 16]

        for lighting_set_name in lighting.keys():
            if lighting_set_name == 'led':
                continue
            ProtocolGenerator.generate_protocol_from_lighting_set(lighting, lighting_set_name)

    @staticmethod
    def generate_protocol_from_lighting_set(lighting, lighting_set):
        logger.info(' Creating protocol %s...' % lighting_set.upper())

        random.seed(22)

        banned = {2: 3, 7: 1}

        subject_set = {}
        subject_set['train'] = []
        subject_set['test'] = []

        temp = copy.deepcopy(lighting[lighting_set])
        while len(subject_set['train']) < (2.0 / 3.0) * len(lighting[lighting_set]):
            i = random.randint(0, len(temp) - 1)
            subject_set['train'].append(temp[i])
            temp[i] = temp[-1]
            temp = temp[:-1]

        temp_led = copy.deepcopy(lighting['led'])
        while lighting_set == 'all' and len(temp_led) > 1:
            i = random.randint(0, len(temp_led) - 1)
            subject_set['train'].append(temp_led[i])
            temp_led[i] = temp_led[-1]
            temp_led = temp_led[:-1]

        while len(temp) > 0:
            subject_set['test'].append(temp[-1])
            temp = temp[:-1]

        while lighting_set == 'all' and len(temp_led) > 0:
            subject_set['test'].append(temp_led[-1])
            temp_led = temp_led[:-1]

        outdir = '/datagrid/personal/spetlrad/hr/db/ecg-fitness/protocols-v2'

        tasks = ['speak', 'row', 'speak-corrupted', 'row-corrupted', 'stepper', 'bike']
        subsets = ['all', 'train', 'test', 'train-train', 'train-validation']
        cam_protocols = {'': ['c920-1', 'c920-2'], '-tripod': ['c920-1'], '-machine': ['c920-2']}
        task_groups = {}
        task_groups['complete'] = [0, 1, 2, 3, 4, 5]
        task_groups['clean'] = [0, 1, 4, 5]
        task_groups['corrupted'] = [2, 3]
        task_groups['both-speak'] = [0, 2]
        task_groups['both-row'] = [1, 3]
        task_groups['single-speak'] = [0]
        task_groups['single-row'] = [1]
        task_groups['single-speak-corrupted'] = [2]
        task_groups['single-row-corrupted'] = [3]
        task_groups['single-stepper'] = [4]
        task_groups['single-bike'] = [5]
        for cam_protocol, cam_names in cam_protocols.items():
            for task_group_name, task_ids in task_groups.items():
                protocol_path = os.path.join(outdir, lighting_set + '-' + task_group_name + cam_protocol)
                if not os.path.isdir(protocol_path):
                    os.makedirs(protocol_path)

                protocol_filename = {}
                protocol_file = {}
                for subset in subsets:
                    protocol_filename[subset] = os.path.join(protocol_path, subset + '.txt')
                    protocol_file[subset] = open(protocol_filename[subset], 'w')

                active_subset = 'train'
                for subject_id in subject_set[active_subset]:
                    for task_id in task_ids:
                        if banned.get(subject_id, None) is not None and banned[subject_id] == task_id:
                            continue
                        for cam in cam_names:
                            protocol_file[active_subset].write('%02d/%02d/%s\n' % (subject_id, task_id + 1, cam))
                            protocol_file['all'].write('%02d/%02d/%s\n' % (subject_id, task_id + 1, cam))

                active_subset = 'test'
                for subject_id in subject_set[active_subset]:
                    for task_id in task_ids:
                        if banned.get(subject_id, None) is not None and banned[subject_id] == task_id:
                            continue
                        for cam in cam_names:
                            protocol_file[active_subset].write('%02d/%02d/%s\n' % (subject_id, task_id + 1, cam))
                            protocol_file['all'].write('%02d/%02d/%s\n' % (subject_id, task_id + 1, cam))

                active_subset = 'train'
                active_subsubset = '-train'
                for subject_counter, subject_id in enumerate(subject_set[active_subset]):
                    for task_id in task_ids:
                        if banned.get(subject_id, None) is not None and banned[subject_id] == task_id:
                            continue
                        if subject_counter >= (2.0 / 3.0) * len(subject_set[active_subset]):
                            active_subsubset = '-validation'
                        for cam in cam_names:
                            protocol_file[active_subset + active_subsubset].write('%02d/%02d/%s\n' % (subject_id, task_id + 1, cam))

                for subset in subsets:
                    protocol_file[subset].close()


class DatasetWorker(object):
    @staticmethod
    def prepare_list_of_files(hr_directory, db_name, subset='all', protocol='all'):
        if 'compressed' in db_name:
            db_name = db_name.split('-')[0]

        db_path = os.path.join(hr_directory, 'db', db_name)

        file = open(os.path.join(db_path, 'protocols', protocol, subset + '.txt'), 'r')
        return [file.replace('\n', '') for file in file.readlines()]

    @staticmethod
    def check_dataset_by_exporting_to_video(db_name, subset='all', fps=0.0, video_width_height=(0, 0)):
        from cmp.nrppg.cnn.dataset.FaceDatasetFFmpegVideo import FaceDatasetFFmpegVideo
        from cmp.nrppg.torch.utils import opencv_colordim_switch
        hr_directory = os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')
        db_path = os.path.join(hr_directory, 'db', db_name)

        height = 192
        width = 128

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(filename=os.path.join(db_path, 'check.avi'), fourcc=fourcc, fps=fps,
                                       frameSize=(width, height))

        file_list = DatasetWorker.prepare_list_of_files(hr_directory, db_name, subset)

        for file in file_list:
            file = file.replace('\n', '')

            bboxes = PureDatasetWorker.load_bboxes(os.path.join(hr_directory, 'db', db_name, 'bbox', file + '.face'))
            faceDB = FaceDatasetFFmpegVideo(os.path.join(db_path, file + '.avi'), None, fps, 500, video_width_height, bboxes)

            frame_count = int(faceDB.length)
            for frame_id in range(frame_count):
                opencv_img = opencv_colordim_switch(faceDB.data[frame_id].transpose(1, 2, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(opencv_img, file.replace('/data', ''), (10, 175), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(opencv_img)

        video_writer.release()

    @staticmethod
    def convert_fase_sequence_to_single_png():
        path = '/datagrid/personal/spetlrad/hr/db/ecg-fitness/'
        files = os.listdir(path + 'ecf-all/')
        dpi = 72
        dpmm = dpi / 2.52
        width_mm = 17
        width_px = int(width_mm * dpmm)
        height_px = int((float(width_px)/16.0)*9.0)
        images_per_line = 10
        line_width_px = width_px * images_per_line
        lines = 200 / images_per_line
        empty_image = np.zeros((height_px * lines, line_width_px, 3), dtype='uint8')
        for file_id, file in enumerate(files):
            im = cv2.imread(path + 'ecf-all/' + file)
            im = cv2.resize(im, (width_px, height_px))
            xpos = file_id % images_per_line
            ypos = int(file_id / images_per_line)
            empty_image[ypos*height_px:(ypos+1)*height_px, xpos*width_px:(xpos+1)*width_px, :] = im

        cv2.imwrite(path + 'all.png', empty_image)

    @staticmethod
    def extract_face_sequence_to_png(db_name, subset, fps, video_width_height, frame_range, desired_video=None):
        from cmp.nrppg.cnn.dataset.FaceDatasetFFmpegVideo import FaceDatasetFFmpegVideo
        from cmp.nrppg.torch.utils import opencv_colordim_switch
        import glob

        hr_directory = os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')
        db_path = os.path.join(hr_directory, 'db', db_name)

        file_list = DatasetWorker.prepare_list_of_files(hr_directory, db_name, subset)

        for file in file_list:
            file = file.replace('\n', '')
            if desired_video is not None and desired_video not in file:
                continue

            if db_name == Dataset.HCI:
                file += '/' + os.path.basename(glob.glob(os.path.join(db_path, file, '*.avi'))[0])[:-4]

            #bboxes = PureDatasetWorker.load_bboxes(os.path.join(hr_directory, 'db', db_name, 'bbox', file + '.face'))
            bboxes = None
            faceDB = FaceDatasetFFmpegVideo(os.path.join(db_path, file + '.avi'), None, fps, 500, video_width_height, bboxes)

            img_path = os.path.join(hr_directory, 'db', db_name, 'intro-check')
            if not os.path.isdir(img_path):
                os.mkdir(img_path)

            frame_count = int(faceDB.length)
            for frame_id in frame_range:
                opencv_img = opencv_colordim_switch(faceDB.data[frame_id].transpose(1, 2, 0))
                cv2.imwrite(os.path.join(img_path, '%s-%010d.png' % (file.replace('/','-'), frame_id)), opencv_img)


    @staticmethod
    def get_extractor_outputs(db_name):
        experiments_directory = DatasetWorker.get_experiments_directory()
        extractor_model_name = '20-03-2018_00-56-42-838138_arch=FaceHRNet09V4ELURGB_lr=1E-04_min-std=1E-01_em-after=1002_batch-size=600' \
                               + '_obj=snr_scale-factor=0E+00_rotation-augment_batch-lightness-jitter=pm50_epoch=301_snr_median_best'
        return os.path.join(experiments_directory, 'runs-%s-eval' % db_name, '%s_%s.data' % (db_name, extractor_model_name))

    @staticmethod
    def get_hr_directory():
        return os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')

    @staticmethod
    def get_experiments_directory():
        return os.path.join(DatasetWorker.get_hr_directory(), 'experiments', 'cnn')

    @staticmethod
    def get_qf(db_name):
        qf = None
        if 'compressed' in db_name:
            qf = 23
        return qf


class ECGFitnessDatasetWorker(DatasetWorker):
    @staticmethod
    def load_ecg(basepath, camera_type, raw=True):
        if camera_type != 'c920' and camera_type != 'flir':
            raise Exception("Wrong camera type.")

        basepath = basepath.replace('/c920-0', '')
        basepath = basepath.replace('/c920-1', '')
        basepath = basepath.replace('/c920-2', '')
        basepath = basepath.replace('/flir', '')
        camera_csv_file_path = os.path.join(basepath, camera_type + '.csv')
        if not os.path.isfile(camera_csv_file_path):
            raise Exception("File %s does not exist." % camera_csv_file_path)

        viatom_raw_csv_file_path = os.path.join(basepath, 'viatom-raw.csv')
        if not os.path.isfile(viatom_raw_csv_file_path):
            raise Exception("File %s does not exist." % viatom_raw_csv_file_path)

        ecg_idxs = np.loadtxt(camera_csv_file_path, skiprows=0, delimiter=',')
        ecg_data_raw = np.loadtxt(viatom_raw_csv_file_path, skiprows=1, delimiter=',')

        if raw is not True:
            ecg_idxs = ecg_idxs[:, 1]
            ecg_data_raw = ecg_data_raw[:, 1]

        return ecg_data_raw, ecg_idxs


class PureDatasetWorker(DatasetWorker):
    @staticmethod
    def load_hr(json_path):
        json_data = json.load(open(json_path))

        keys_count = len(json_data['/Image'])
        keys_count_full = len(json_data['/FullPackage'])
        hr = np.zeros((int(keys_count)), dtype='int32')
        current_image_key = 0
        current_json_key = 0
        while current_json_key < len(json_data['/FullPackage']):
            if json_data['/FullPackage'][current_json_key]['Timestamp'] <= json_data['/Image'][current_image_key]['Timestamp'] <= \
                    json_data['/FullPackage'][current_json_key + 1]['Timestamp'] \
                    or (current_json_key == 0 and json_data['/Image'][current_image_key]['Timestamp'] <=
                        json_data['/FullPackage'][current_json_key + 1]['Timestamp']):
                hr[current_image_key] = json_data['/FullPackage'][current_json_key]['Value']['pulseRate']
                current_image_key += 1
                if current_image_key == keys_count:
                    break
            else:
                current_json_key += 1
                if current_json_key + 1 == keys_count_full:
                    break
        hr[hr == 0] = hr[hr != 0].mean()

        return hr

    @staticmethod
    def load_bboxes(bbox_path):
        return np.loadtxt(bbox_path, delimiter=' ')[:, 1:]

    @staticmethod
    def check_dataset_by_exporting_to_video(db_name='pure', subset='all', fps=0.0, video_width_height=(0, 0)):
        DatasetWorker.check_dataset_by_exporting_to_video(db_name, subset, 30.0, (640, 480))

    @staticmethod
    def prepare_hard_negative_face_dataset(hr_directory, subset, batch_size, transform=None, allowed_list=None):
        db_name = 'pure'
        file_list = PureDatasetWorker.prepare_list_of_files(hr_directory, db_name, subset)

        hard_negative_file_list = []
        y_bpm = []
        bboxes = []
        fps = []
        video_dimensions = []
        for file in file_list:
            if allowed_list is not None:
                skip = True
                for allowed in allowed_list:
                    if allowed in file:
                        skip = False
                        break
                if skip:
                    continue

            hard_negative_file_list.append(os.path.join(hr_directory, 'db', db_name, file + '.avi'))
            y_bpm.append(PureDatasetWorker.load_hr(os.path.join(hr_directory, 'db', 'pure', 'gt', file + '.json')))
            bboxes.append(PureDatasetWorker.load_bboxes(os.path.join(hr_directory, 'db', 'pure', 'bbox', file + '.face')))
            fps.append(30.0)
            video_dimensions.append((640, 480))

        from cmp.nrppg.cnn.dataset.FaceDatasetFFmpegVideoList import FaceDatasetFFmpegVideoList
        return FaceDatasetFFmpegVideoList(hard_negative_file_list, y_bpm, fps, batch_size, video_dimensions, bboxes_list=bboxes,
                                          transform=transform)


if __name__ == '__main__':
    #ProtocolGenerator.generate_protocols()
    # DatasetWorker.check_dataset_by_exporting_to_video('ecg-fitness', 'all', 30.0, (1920, 1080))

    DatasetWorker.convert_fase_sequence_to_single_png()
    exit(0)

    DatasetWorker.extract_face_sequence_to_png(Dataset.ECG_FITNESS,
                                               'all', 30.0, (1920, 1080),
                                               range(2, 3))
    exit(0)
    for second in [0,30, 60]:
        #DatasetWorker.extract_face_sequence_to_png(Dataset.PURE,
        #                                           'all', 30.0, (640, 480),
        #                                           range(30 * second + 1, 30 * second + 2),
        #                                           '01-01')
        #DatasetWorker.extract_face_sequence_to_png(Dataset.COHFACE,
        #                                           'all', 20.0, (640, 480),
        #                                           range(20 * second + 1, 20 * second + 2),
        #                                           '1/0')
        #DatasetWorker.extract_face_sequence_to_png(Dataset.COHFACE,
        #                                           'all', 20.0, (640, 480),
        #                                           range(20 * second + 1, 20 * second + 2),
        #                                           '1/2')
        DatasetWorker.extract_face_sequence_to_png(Dataset.HCI,
                                                   'all', 61.0, (780, 580),
                                                   range(61 * second + 1, 61 * second + 2),
                                                   '2734')
