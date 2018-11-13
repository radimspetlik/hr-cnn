import json
import numpy as np
import cv2
import os


class DatasetWorker(object):
    @staticmethod
    def prepare_list_of_files(hr_directory, db_name, subset='all', protocol='all'):
        db_path = os.path.join(hr_directory, 'db', db_name)

        file = open(os.path.join(db_path, 'protocols', protocol, subset + '.txt'), 'r')
        return [file.replace('\n', '') for file in file.readlines()]


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
    def check_dataset_by_exporting_to_video(subset='all'):
        from cmp.nrppg.cnn.dataset.FaceDatasetFFmpegVideo import FaceDatasetFFmpegVideo
        from cmp.nrppg.torch.utils import opencv_colordim_switch
        hr_directory = os.path.join('/datagrid', 'personal', 'spetlrad', 'hr')
        db_name = 'pure'
        db_path = os.path.join(hr_directory, 'db', db_name)

        height = 192
        width = 128

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(filename=os.path.join(db_path, 'check.avi'), fourcc=fourcc, fps=30.00,
                                       frameSize=(width, height))

        file_list = PureDatasetWorker.prepare_list_of_files(hr_directory, db_name, subset)

        for file in file_list:
            file = file.replace('\n', '')

            bboxes = PureDatasetWorker.load_bboxes(os.path.join(hr_directory, 'db', 'pure', 'bbox', file + '.face'))
            faceDB = FaceDatasetFFmpegVideo(os.path.join(db_path, file + '.avi'), None, 30.0, 500, bboxes)

            frame_count = int(faceDB.length)
            for frame_id in range(frame_count):
                opencv_img = opencv_colordim_switch(faceDB.data[frame_id].transpose(1, 2, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(opencv_img, file.replace('/data', ''), (10, 175), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(opencv_img)

        video_writer.release()

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
    PureDatasetWorker.check_dataset_by_exporting_to_video()
