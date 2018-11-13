import torch.utils.data as data
import numpy as np
import h5py
import cv2
import logging

__logging_format__ = '[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("extract_log")


class FaceDatasetHdf5(data.Dataset):
    def __init__(self, X_path, y, batch_size, train=True, skip_partitioning=False, rgb=False):
        self.train = train
        self.skip_partitioning = skip_partitioning
        self.batch_size = float(batch_size)
        self.y = y
        self.rgb = rgb

        self.last_index_div = 0

        self.hdf5_dataset = h5py.File(X_path, 'r')

        # - 1 accounts for fps key
        if self.y is None:
            self.y = np.zeros((len(self.hdf5_dataset.keys()) - 1))
        if self.y.size == 1:
            self.y = np.ones((len(self.hdf5_dataset.keys()) - 1)) * self.y
        self.length = min(len(self.hdf5_dataset.keys()) - 1, self.y.size)
        self.train_length = int(np.floor((self.length / self.batch_size) * (2.0 / 3.0)) * self.batch_size)

        self.shift = 0

        if not self.skip_partitioning:
            self.test_length = self.length - self.train_length

            self.length = self.test_length
            self.shift = self.train_length

            if self.train:
                self.length = self.train_length

    def __get_data(self, index):
        key = '%10.0d' % index
        return self.hdf5_dataset[key].value

    def get_fps(self, idx):
        idx += self.shift

        return float(self.hdf5_dataset['fps'].value)

    def get_shift(self):
        return self.shift

    def __getitem__(self, index):
        index += self.shift

        data = []
        try:
            data = self.hdf5_dataset['%010.d' % (index + 1)].value.astype('float32')
        except KeyError as e:
            logger.warn("Didn't find key %010.d in the hdf5 file." % (index + 1))
            data = np.zeros((3, 128, 192), dtype='float32')

        data = cv2.resize(data.transpose((1, 2, 0)), (192, 128)).transpose((2, 0, 1))
        if not self.rgb:
            data = data[1, :, :][np.newaxis, :, :]

        data = data.transpose((0, 2, 1))

        if data is not None:
            target = self.y[index].astype('float32')
        else:
            logger.warn("Didn't find key %010.d in the hdf5 file." % (index + 1))
            target = np.array([0.0]).astype('float32')

        return data, target / 60.0

    def __len__(self):
        return self.length