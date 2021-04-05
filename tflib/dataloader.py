import numpy as np
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import os
from config import Config

opt = Config().parse()


def data_generator(batch_size, data, label):
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(label)

        for i in range(int(len(data) / batch_size)):
            yield (data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size])

    return get_epoch


def pad_zeros(input, margin=4):
    output = np.zeros((input.shape[0] + 2 * margin, input.shape[1] + 2 * margin, input.shape[2]))
    row_offset = margin
    col_offset = margin
    output[row_offset:input.shape[0] + row_offset, col_offset:input.shape[1] + col_offset, :] = input
    return output


def load(batch_size=opt.BATCH_SIZE, dataset=opt.DATASET, data_dir=opt.DATA_DIR, test_ratio=opt.TEST_RATIO,
         window_size=opt.WINDOW_SIZE, channel=opt.CHANNEL, use_pca=opt.use_PCA):
    if dataset == 'Salinas':
        # Full-size data.
        data = loadmat(os.path.join(data_dir, 'Salinas_corrected.mat'))['salinas_corrected']
        '''
        We have tried to employ SuperPCA.
        To load data reduced by SuperPCA, just put the reduced data by SuperPCA into ./dataset.
        The channel depends on what you use for SuperPCA.
        '''
        if channel == 5 and not use_pca:
            data_in = loadmat(os.path.join(data_dir, 'Salinas_DR.mat'))['dataDR']
        # Use full-size data.
        elif channel == 204 and not use_pca:
            data_in = data
        # else:
        #     raise NotImplementedError
        labels = loadmat(os.path.join(data_dir, 'Salinas_gt.mat'))['salinas_gt']
    elif dataset == 'Indian':
        # Full-size data.
        data = loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        '''
        We have tried to employ SuperPCA. 
        To load data reduced by SuperPCA, just put the reduced data by SuperPCA into ./dataset.
        The channel depends on what you use for SuperPCA.
        '''
        if channel == 5 and not use_pca:
            data_in = loadmat(os.path.join(data_dir, 'Indian_DR.mat'))['dataDR']
        # Use full-size data.
        elif channel == 200 and not use_pca:
            data_in = data
        # else:
        #     raise NotImplementedError
        labels = loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif dataset == 'PaviaU':
        # Full-size data.
        data = loadmat(os.path.join(data_dir, 'PaviaU.mat'))['paviaU']
        '''
        We have tried to employ SuperPCA. 
        To load data reduced by SuperPCA, just put the reduced data by SuperPCA into ./dataset.
        The channel depends on what you use for SuperPCA.
        '''
        if channel == 5 and not use_pca:
            data_in = loadmat(os.path.join(data_dir, 'PaviaU_DR.mat'))['dataDR']
        # Use full-size data.
        elif channel == 103 and not use_pca:
            data_in = data
        # else:
        #     raise NotImplementedError
        labels = loadmat(os.path.join(data_dir, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise NotImplementedError

    '''
    Apply PCA to reduce dimensionality. 
    '''
    if use_pca:
        data_pca = np.reshape(data, (-1, data.shape[-1]))
        pca = PCA(n_components=channel, whiten=True)
        data_pca = pca.fit_transform(data_pca)
        data_in = np.reshape(data_pca, (data.shape[0], data.shape[1], channel))

    else:
        '''
        Create patches with window_size for each pixel, with zero padding size (window_size - 1) / 2). 
        The label is the central pixel's.
        '''
        margin = int((window_size - 1) / 2)
        data_padded = pad_zeros(data_in, margin=margin)
        # Split patches.
        data_patches = np.zeros((data.shape[0] * data.shape[1], window_size, window_size, data_in.shape[-1]))
        label_patches = np.zeros((data.shape[0] * data.shape[1]))
        patch_index = 0
        for row in range(margin, data_padded.shape[0] - 2 * margin):
            for col in range(margin, data_padded.shape[1] - 2 * margin):
                patch = data_padded[row - margin:row + margin + 1, col - margin:col + margin + 1]
                data_patches[patch_index, :, :, :] = patch
                label_patches[patch_index] = labels[row - margin, col - margin]
                patch_index = patch_index + 1
        # Remove zero labels.
        data_patches = data_patches[label_patches > 0, :, :, :]
        data_patches = data_patches.reshape(data_patches.shape[0], -1)
        label_patches = label_patches[label_patches > 0]
        label_patches -= 1

        train_data, test_data, train_label, test_label = train_test_split(data_patches, label_patches,
                                                                          test_size=test_ratio,
                                                                          random_state=345,
                                                                          stratify=label_patches)
        train_niter_epoch = int(len(train_data) / (batch_size * (opt.CRITIC_ITERS + 1)))

        return (data_generator(batch_size, train_data, train_label), train_niter_epoch,
                data_generator(batch_size, test_data, test_label))

