import numpy as np
from scipy.io import loadmat
import os
from config import Config

opt = Config().parse()


def data_generator(batch_size, data, label):
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)

        for i in range(int(len(data) / batch_size)):
            yield (data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size])

    return get_epoch


def pad_zeros(input, margin=4):
    output = np.zeros((input.shape[0] + 2 * margin, input.shape[1] + 2 * margin, input.shape[2]))
    row_offset = margin
    col_offset = margin
    output[row_offset:input.shape[0] + row_offset, col_offset:input.shape[1] + col_offset, :] = input
    return output


def load_map(batch_size=opt.BATCH_SIZE, dataset=opt.DATASET, data_dir=opt.DATA_DIR, window_size=opt.WINDOW_SIZE):
    if dataset == 'Salinas':
        data = loadmat(os.path.join(data_dir, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = loadmat(os.path.join(data_dir, 'Salinas_gt.mat'))['salinas_gt']
    elif dataset == 'Indian':
        data = loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif dataset == 'PaviaU':
        data = loadmat(os.path.join(data_dir, 'PaviaU.mat'))['paviaU']
        labels = loadmat(os.path.join(data_dir, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise NotImplementedError

    '''
    Create patches with window_size for each pixel, with zero padding size (window_size - 1) / 2. 
    The label is the central pixel's.
    '''
    margin = int((window_size - 1) / 2)
    data_padded = pad_zeros(data, margin=margin)
    # Split patches.
    if dataset == 'Salinas':
        data_patches = np.zeros((data.shape[0] * data.shape[1] + 3, window_size, window_size, data.shape[-1]))
        label_patches = np.zeros((data.shape[0] * data.shape[1] + 3))
    elif dataset == 'Indian':
        data_patches = np.zeros((data.shape[0] * data.shape[1] + 31, window_size, window_size, data.shape[-1]))
        label_patches = np.zeros((data.shape[0] * data.shape[1] + 31))
    elif dataset == 'PaviaU':
        data_patches = np.zeros((data.shape[0] * data.shape[1] + 24, window_size, window_size, data.shape[-1]))
        label_patches = np.zeros((data.shape[0] * data.shape[1] + 24))
    patch_index = 0
    for row in range(margin, data_padded.shape[0] - margin):
        for col in range(margin, data_padded.shape[1] - margin):
            patch = data_padded[row - margin:row + margin + 1, col - margin:col + margin + 1]
            data_patches[patch_index, :, :, :] = patch
            label_patches[patch_index] = labels[row - margin, col - margin]
            patch_index = patch_index + 1

    data_patches = data_patches.reshape(data_patches.shape[0], -1)
    label_patches -= 1

    return (data, labels, data_generator(batch_size, data_patches, label_patches))


