import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-GPU', type=int, default=7, help='which gpu device to use')

        self.parser.add_argument('-MODE', default='local', choices=['local', 'global'], help='type of loss')
        self.parser.add_argument('-MODE_GAC', default='gan', choices=['gan', 'wgan', 'wgan_gp'], help='type of loss')
        self.parser.add_argument('-MODE_BGAC', default='ali', choices=['ali', 'wali', 'wali_gp'], help='type of loss')

        self.parser.add_argument('-DATASET', default='Salinas', choices=['Salinas', 'Indian', 'PaviaU'],
                                 help='which data set for experiment')
        self.parser.add_argument('-DIM', type=int, default=64, help='model dimensionality')
        self.parser.add_argument('-LATENT_DIM', type=int, default=100, help='hidden dimensionality')
        self.parser.add_argument('-CHANNEL', type=int, default=204, help='input channels')
        self.parser.add_argument('-WINDOW_SIZE', type=int, default=9,
                                 help='size of training/testing patches')
        self.parser.add_argument('-OUTPUT_DIM', type=int, default=16524,
                                 help='number of pixels of output')
        self.parser.add_argument('-N_CLS', type=int, default=16, choices=[16, 16, 9],
                                 help='how many class in the training data')
        self.parser.add_argument('-N_GM', type=int, default=16,
                                 help='how many mixed ingredients in gaussian mixture model')

        self.parser.add_argument('-ITERS', type=int, default=100000, help='how many generator iterations to train for')
        self.parser.add_argument('-CRITIC_ITERS', type=int, default=1,
                                 help='Number of discriminator training steps for each generator training step')
        self.parser.add_argument('-BATCH_SIZE', type=int, default=64, help='training batch size')
        self.parser.add_argument('-BN_FLAG', type=bool, default=True, help='do batch normalization or not')
        self.parser.add_argument('-DR_RATE', type=float, default=.2, help='dropout rate')

        self.parser.add_argument('-LR', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('-BETA1', type=float, default=.5, help='')
        self.parser.add_argument('-BETA2', type=float, default=.999, help='')
        self.parser.add_argument('-TEMP', type=float, default=.1, help='molecule of sample_gumbel shape')

        self.parser.add_argument('-use_PCA', type=bool, default=False,
                                 help='use PCA or not, or use SuperPCA')
        self.parser.add_argument('-with_CORR', type=bool, default=False,
                                 help='use correlation loss or not')
        self.parser.add_argument('-with_REC', type=bool, default=False,
                                 help='use reconstruction loss or not')

        self.parser.add_argument('-DISTANCE_TYPE', default='l2', help='type of reconstruction loss')
        self.parser.add_argument('-EMBEDDING_TYPE', default='LATENT', choices=['LATENT', '10', 'NONE'],
                                 help='embedding type')

        self.parser.add_argument('-TEST_RATIO', type=float, default=.9, help='ratio of training data in the full set')
        self.parser.add_argument('-TEST_TH', type=float, default=.8, help='threshold of training accuracy for testing')

        self.parser.add_argument('-DATA_DIR', default='./dataset/', help='directory to load data')
        self.parser.add_argument('-TRANSFORMED_DATA_DIR', default='./transformed_data',
                                 help='directory to save transformed data')
        self.parser.add_argument('-OUTPUT', default='./result/', help='directory to save results')


    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        return self.opt
