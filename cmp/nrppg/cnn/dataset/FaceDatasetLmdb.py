import lmdb
import numpy as np
from cmp.nrppg.cnn.dataset.FaceDataset import FaceDataset


class FaceDatasetLmdb(FaceDataset):
    def __init__(self, X_lmdb_path, y_lmdb_path, batch_size, train=True, skip_partitioning=False, rgb=False, transform=None):
        self.transform = transform
        self.rgb = rgb
        self.train = train
        self.skip_partitioning = skip_partitioning
        self.batch_size = float(batch_size)

        self.last_index_div = 0

        self.data_env = lmdb.open(X_lmdb_path, readonly=True)
        self.label_env = lmdb.open(y_lmdb_path, readonly=True)

        self.data_txn = self.data_env.begin()
        self.data_cursor = self.data_txn.cursor()

        self.length = np.fromstring(self.data_cursor.get('frame_count'.encode('ascii')), dtype='int32')

        if self.skip_partitioning:
            self.shift = 0
        else:
            self.train_length = int(np.floor((self.length / self.batch_size) * (2.0 / 3.0)) * self.batch_size)
            # handle identity aware splitting
            while self.data_cursor.get('fps-{:08}'.format(self.train_length).encode('ascii'), default=None) is None:
                self.train_length += int(self.batch_size)

            self.test_length = self.length - self.train_length

            if self.train:
                self.length = self.train_length
                self.shift = 0
            else:
                self.length = self.test_length
                self.shift = self.train_length

        self.label_txn = self.label_env.begin()
        self.label_cursor = self.label_txn.cursor()

        FaceDataset.__init__(self, self.length, self.transform)

    def __get_height(self, cursor, index):
        search_index = int(index)
        string_data = cursor.get(('height-%08d' % search_index).encode('ascii'))
        while string_data is None:
            search_index -= int(self.batch_size)
            string_data = cursor.get(('height-%08d' % search_index).encode('ascii'))

            if search_index < 0:
                raise KeyError('Key cannot be lower than 0, started at %d...' % index)

        return int(np.fromstring(string_data, dtype='int32').round())

    def __get_data(self, cursor, index):
        key = '{:08}'.format(index)
        string_data = cursor.get(key.encode('ascii'))
        return string_data

    def get_fps_and_regularization_factor(self, idx):
        idx += self.shift

        fps_string = self.data_cursor.get('fps-{:08}'.format(idx).encode('ascii'), default=None)
        regularization_factor_string = self.data_cursor.get('regularization_factor-{:08}'.format(idx).encode('ascii'))
        while fps_string is None:
            idx -= int(self.batch_size)
            fps_string = self.data_cursor.get('fps-{:08}'.format(idx).encode('ascii'), default=None)
            regularization_factor_string = self.data_cursor.get('regularization_factor-{:08}'.format(idx).encode('ascii'))

        regularization_factor = float(np.fromstring(regularization_factor_string, dtype='float64').mean())

        return float(np.fromstring(fps_string, dtype='float64').mean()), regularization_factor

    def get_shift(self):
        return self.shift

    def get_original_and_transformed_im(self, idx=0):
        orig_data = self.get_im_data(idx)

        transformed_data = orig_data
        if self.transform is not None:
            transformed_data = self.do_transforms(orig_data)

        return orig_data.astype('uint8'), transformed_data.astype('uint8')

    def get_im_data(self, idx):
        string_data = self.__get_data(self.data_cursor, idx)
        if string_data is None:
            print('missing %d' % idx)

        height = self.__get_height(self.data_cursor, int(idx / self.batch_size) * int(self.batch_size))
        flat_data = np.fromstring(string_data, dtype='uint8')
        width = int((flat_data.size / 3) / height)
        flat_data = flat_data.reshape(height, width, 3).transpose((2, 1, 0))
        if not self.rgb:
            flat_data = flat_data[1, :, :].reshape(1, width, height)

        return flat_data.astype('float32')

    def __getitem__(self, index):
        index += self.shift

        data = self.get_im_data(index)
        if self.transform is not None:
            data = self.do_transforms(data)

        string_data = self.__get_data(self.label_cursor, index)

        if string_data is not None:
            y = np.fromstring(string_data, dtype='int32')
            target = y[0].astype('float32')
        else:
            raise ValueError('Missing ground truth in the database for key %d...' % index)

        return data, target / 60.0
