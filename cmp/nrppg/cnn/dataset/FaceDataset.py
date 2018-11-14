import torch.utils.data as data
import torchvision as vision


class FaceDataset(data.Dataset):
    def __init__(self, length, transform=None):
        self.length = length
        self.transform = transform

    def do_transforms(self, data):
        toPIL = vision.transforms.ToPILImage()
        toTensor = vision.transforms.ToTensor()
        data = toTensor(self.transform(toPIL(data.astype('uint8').transpose((1, 2, 0))))) * 255
        return data.numpy()

    def get_original_and_transformed_im(self, idx=0):
        orig_data = self.get_im_data(idx)

        transformed_data = orig_data
        if self.transform is not None:
            transformed_data = self.do_transforms(orig_data)

        return orig_data.astype('uint8'), transformed_data.astype('uint8')

    def get_im_data(self, idx):
        raise NotImplementedError

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):
        raise NotImplementedError
