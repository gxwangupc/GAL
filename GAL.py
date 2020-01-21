import numpy as np
import tensorflow as tf
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import shutil
import spectral

import tflib as lib
from tflib.ops.linear import Linear
from tflib.ops.conv2d import Conv2D
from tflib.ops.batchnorm import Batchnorm
from tflib.ops.deconv2d import Deconv2D
from tflib.ops.embedding import Embedding
from tflib.dataloader import load
from tflib.dataloader_map import load_map
from tflib.RAdam import RAdamOptimizer
from tflib.config import Config

import os
import sys

'''Get current dictionary.'''
sys.path.append(os.getcwd())

'''Get namespace.'''
opt = Config().parse()

'''Specify which GPU to use.'''
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPU)
print("using gpu {}".format(opt.GPU))


"""
Ceate log dictionaries for the current executed code.
The created dictionaries are in './result/' if opt.OUTPUT is set to './result/'.
"""
filename_script = os.path.basename(os.path.realpath(__file__))
outf = os.path.join(opt.OUTPUT, os.path.splitext(filename_script)[0])
outf += '.MODE-'
outf += opt.MODE
outf += '.N_GM-'
outf += str(opt.N_GM)
outf += '.DATASET-'
outf += str(opt.DATASET)
if not os.path.exists(outf):
    os.makedirs(outf)
    os.makedirs(os.path.join(outf, 'ckpt'))
file_name = 'log' + '.txt'
logfile = os.path.join(outf, file_name)
shutil.copy(os.path.realpath(__file__), os.path.join(outf, filename_script))


"""
models
"""
'''Prior for k.'''
PI = tf.constant(np.asarray([1. / opt.N_GM, ] * opt.N_GM, dtype=np.float32))
prior_k = tf.distributions.Categorical(probs=PI)

'''Calculate Pearson correlation loss.'''
def correlation(x, y):
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)
    x = tf.math.l2_normalize(x, axis=1)
    y = tf.math.l2_normalize(y, axis=1)
    return tf.reduce_sum(x * y, axis=1, keepdims=True)


'''Calculate Reconstruction loss: Mean Square Error.'''
def l2(x, y):
    return tf.reduce_mean(tf.pow(x - y, 2))


'''An alternative Reconstruction loss.'''
def l1(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def distance(x, y, d_type):
    xs = tf.shape(x)
    x = tf.reshape(x, [-1, xs[-1]])
    ys = tf.shape(y)
    y = tf.reshape(y, [-1, ys[-1]])
    if d_type is 'l1':
        return l1(x,y)
    elif d_type is 'l2':
        return l2(x,y)


'''Gumbel-Softmax estimators.'''
def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    u = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


def leakyrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


'''
Loss function.
'''
def overall_loss(disc_fake_list, disc_real_list, gen_params, disc_params, lr=2e-4, beta1=0.5, beta2=.999, s_f=None,
                 rec_loss=None, cls_real_loss=None, cls_fake_loss=None, corr_loss=None, adam=False):
    gen_loss = 0
    disc_loss = 0
    '''local mode.'''
    if isinstance(disc_fake_list, list):
        for disc_fake, disc_real in zip(disc_fake_list, disc_real_list):
            gen_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake,
                labels=tf.ones_like(disc_fake)
            ))
            gen_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_real,
                labels=tf.zeros_like(disc_real)
            ))

            disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake,
                labels=tf.zeros_like(disc_fake)
            ))
            disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_real,
                labels=tf.ones_like(disc_real)
            ))
        gen_loss /= len(disc_fake_list)
        disc_loss /= len(disc_fake_list)
    else:
        '''global mode'''
        gen_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake_list,
            labels=tf.ones_like(disc_fake_list)
        ))
        gen_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real_list,
            labels=tf.zeros_like(disc_real_list)
        ))

        disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake_list,
            labels=tf.zeros_like(disc_fake_list)
        ))
        disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real_list,
            labels=tf.ones_like(disc_real_list)
        ))

    if s_f is not None:
        gen_loss += s_f
    if rec_loss is not None:
        gen_loss += rec_loss
    if (cls_real_loss is not None) and (cls_fake_loss is not None):
        gen_loss += cls_real_loss + cls_fake_loss
        disc_loss += cls_real_loss + cls_fake_loss

    if corr_loss is not None:
        gen_loss -= 0.5 * corr_loss
        disc_loss -= 0.5 * corr_loss

    '''Use Adam optimizer.'''
    if adam:
        gen_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2
        ).minimize(gen_loss, var_list=gen_params)

        disc_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2
        ).minimize(disc_loss, var_list=disc_params)
    else:
        '''Use RAdam optimizer. We have attempted to use but seen apparent performance decline.'''
        gen_train_op = RAdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2
        ).minimize(gen_loss, var_list=gen_params)

        disc_train_op = RAdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2
        ).minimize(disc_loss, var_list=disc_params)

    return gen_loss, disc_loss, gen_train_op, disc_train_op


'''
hyper_generator: G1.
Actually, There is an embedding layer that learns a matrix Mu and
conducts a simple matrix multiplication to turn positive integers, i.e., hyper_k, 
into dense vectors of a fixed size LATENT_DIM.
'''
def hyper_generator(hyper_k, hyper_noise):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(opt.N_GM, opt.LATENT_DIM)).astype('float32'))
    noise = tf.add(tf.matmul(tf.cast(hyper_k, tf.float32), com_mu), hyper_noise)
    return noise


'''
hyper_extractor: E2.
'''
def hyper_extractor(latent_z):
    com_mu = lib.param('Generator.Hyper.Mu', np.random.normal(size=(opt.N_GM, opt.LATENT_DIM)).astype('float32'))
    com_logits = -.5 * tf.reduce_sum(tf.pow((tf.expand_dims(latent_z, axis=1) - tf.expand_dims(com_mu, axis=0)), 2),
                                     axis=-1) + tf.expand_dims(tf.math.log(PI), axis=0)
    k = tf.nn.softmax((com_logits + sample_gumbel(tf.shape(com_logits))) / opt.TEMP)

    return com_logits, k


'''
generator: G2.
'''
def generator(noise, label):
    # We have implemented three types of combination for noise and label. We used the first finally.
    if opt.EMBEDDING_TYPE == 'LATENT':
        label_emb = Embedding('Generator.Embedding', opt.N_CLS, opt.LATENT_DIM, label)
        gen_input = tf.multiply(label_emb, noise)
        output = Linear('Generator.Input', opt.LATENT_DIM, 3 * 3 * 8 * opt.DIM, gen_input)
    elif opt.EMBEDDING_TYPE == '10':
        label_emb = Embedding('Generator.Embedding', opt.N_CLS, 10, label)
        gen_input = tf.concat([label_emb, noise], 1)
        output = Linear('Generator.Input', opt.LATENT_DIM + 10, 3 * 3 * 8 * opt.DIM, gen_input)
    elif opt.EMBEDDING_TYPE == 'NONE':
        gen_input = tf.concat([tf.one_hot(indices=label, depth=opt.N_CLS), noise], 1)
        output = Linear('Generator.Input', opt.LATENT_DIM + opt.N_CLS, 3 * 3 * 8 * opt.DIM, gen_input)
    else:
        raise NotImplementedError

    if opt.BN_FLAG:
        output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8 * opt.DIM, 3, 3])

    '''Four Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11, 13.'''
    if opt.WINDOW_SIZE == 7:
        output = Deconv2D('Generator.2', 8 * opt.DIM, 4 * opt.DIM, 3, output, padding='SAME', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.3', 4 * opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.4', 2 * opt.DIM, opt.CHANNEL, 3, output, padding='VALID', stride=1)
        output = tf.tanh(output)

    elif opt.WINDOW_SIZE == 9:
        output = Deconv2D('Generator.2', 8 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.3', 4 * opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.4', 2 * opt.DIM, opt.CHANNEL, 3, output, padding='VALID', stride=1)
        output = tf.tanh(output)

    elif opt.WINDOW_SIZE == 11:
        output = Deconv2D('Generator.2', 8 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.3', 4 * opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.4', 2 * opt.DIM, opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN4', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.5', opt.DIM, opt.CHANNEL, 3, output, padding='VALID', stride=1)
        output = tf.tanh(output)

    elif opt.WINDOW_SIZE == 13:
        output = Deconv2D('Generator.2', 8 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.3', 8 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.4', 4 * opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN4', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.5', 2 * opt.DIM, opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Generator.BN5', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = Deconv2D('Generator.6', opt.DIM, opt.CHANNEL, 3, output, padding='VALID', stride=1)
        output = tf.tanh(output)
    else:
        raise NotImplementedError

    return tf.reshape(output, [-1, opt.OUTPUT_DIM])


'''
extractor: E1.
'''
def extractor(inputs):
    '''Four Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11, 13.'''
    if opt.WINDOW_SIZE == 7:
        output = tf.reshape(inputs, [-1, opt.CHANNEL, 7, 7])

        output = Conv2D('Extractor.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        output = leakyrelu(output)

        output = Conv2D('Extractor.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN1', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='SAME', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN2', [0, 2, 3], output)
        output = leakyrelu(output)

    elif opt.WINDOW_SIZE == 9:
        output = tf.reshape(inputs, [-1, opt.CHANNEL, 9, 9])

        output = Conv2D('Extractor.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        output = leakyrelu(output)

        output = Conv2D('Extractor.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN1', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN2', [0, 2, 3], output)
        output = leakyrelu(output)

    elif opt.WINDOW_SIZE == 11:
        output = tf.reshape(inputs, [-1, opt.CHANNEL, 11, 11])

        output = Conv2D('Extractor.1', opt.CHANNEL, opt.DIM, 3, output, padding='VALID', stride=1)
        output = leakyrelu(output)

        output = Conv2D('Extractor.2', opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN1', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.3', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN2', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.4', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN3', [0, 2, 3], output)
        output = leakyrelu(output)

    elif opt.WINDOW_SIZE == 13:
        output = tf.reshape(inputs, [-1, opt.CHANNEL, 13, 13])

        output = Conv2D('Extractor.1', opt.CHANNEL, opt.DIM, 3, output, padding='VALID', stride=1)
        output = leakyrelu(output)

        output = Conv2D('Extractor.2', opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN1', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.3', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN2', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.4', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN3', [0, 2, 3], output)
        output = leakyrelu(output)

        output = Conv2D('Extractor.5', 8 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
        if opt.BN_FLAG:
            output = Batchnorm('Extractor.BN4', [0, 2, 3], output)
        output = leakyrelu(output)

    else:
        raise NotImplementedError

    output = tf.reshape(output, [-1, 3 * 3 * 8 * opt.DIM])
    output = Linear('Extractor.Output', 3 * 3 * 8 * opt.DIM, opt.LATENT_DIM, output)

    return tf.reshape(output, [-1, opt.LATENT_DIM])


'''
discriminators: D1 & D2 in local; D in global.
'''
if opt.MODE is 'local':
    '''hyper_discriminator: D2'''
    def hyper_discriminator(z, k):
        output = tf.concat([z, k], 1)
        '''128 neurons'''
        output = Linear('Discriminator.HyperInput', opt.LATENT_DIM + opt.N_GM, 128, output)
        output = leakyrelu(output)
        output = tf.layers.dropout(output, rate=1. - keep_prob)

        output = Linear('Discriminator.Hyper2', 128, 64, output)
        output = leakyrelu(output)
        output = tf.layers.dropout(output, rate=1. - keep_prob)

        output = Linear('Discriminator.HyperOutput', 64, 1, output)
        return tf.reshape(output, [-1])


    '''discriminator: D1'''
    def discriminator(x, z):
        '''Four Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11, 13.'''
        if opt.WINDOW_SIZE == 7:
            output = tf.reshape(x, [-1, opt.CHANNEL, 7, 7])

            output = Conv2D('Discriminator.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='SAME', stride=1)
            output = leakyrelu(output)

        elif opt.WINDOW_SIZE == 9:
            output = tf.reshape(x, [-1, opt.CHANNEL, 9, 9])

            output = Conv2D('Discriminator.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

        elif opt.WINDOW_SIZE == 11:
            output = tf.reshape(x, [-1, opt.CHANNEL, 11, 11])

            output = Conv2D('Discriminator.1', opt.CHANNEL, opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.2', opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.3', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)

            output = Conv2D('Discriminator.4', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
        else:
            raise NotImplementedError

        output = tf.reshape(output, [-1, 3 * 3 * 8 * opt.DIM])
        '''treat D as an extractor, this is the output'''
        de_output = Linear('Discriminator.de1', 3 * 3 * 8 * opt.DIM, opt.LATENT_DIM, output)

        # '''
        # Scheme 1.
        # We have attempted the use of this version firstly.
        # This scheme has not taken latent representations into use for classification.
        # The effect of extractors is only embodied in the parameters of filters for x.
        # '''
        # '''for classification'''
        # cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, opt.N_CLS, output)
        # # cls_output = Linear('Discriminator.c1', 3 * 3 * 4 * opt.DIM, 256, output)
        # # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
        # # cls_output = tf.nn.softmax(cls_output)
        #
        # z_output = Linear('Discriminator.z1', opt.LATENT_DIM, 512, z)
        # # z_output = Linear('Discriminator.z1', opt.LATENT_DIM, 512, z)
        # z_output = leakyrelu(z_output)
        #
        # output = tf.concat([output, z_output], 1)
        # output = Linear('Discriminator.zx1', 3 * 3 * 8 * opt.DIM + 512, 128, output)
        # # output = Linear('Discriminator.zx1', 3 * 3 * 4 * opt.DIM + 512, 512, output)
        # output = leakyrelu(output)
        #
        # output = Linear('Discriminator.Output', 128, 1, output)
        # # output = Linear('Discriminator.Output', 512, 1, output)

        '''Scheme 2.'''
        z_output = Linear('Discriminator.z1', opt.LATENT_DIM, 512, z)
        z_output = leakyrelu(z_output)

        output = tf.concat([output, z_output], 1)
        output = Linear('Discriminator.zx1', 3 * 3 * 8 * opt.DIM + 512, 128, output)
        # output = Linear('Discriminator.zx1', 3 * 3 * 8 * opt.DIM + 512, 512, output)
        output = leakyrelu(output)

        cls_output = Linear('Discriminator.c1', 128, opt.N_CLS, output)
        # cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, 256, output)
        # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
        # cls_output = tf.nn.softmax(cls_output)

        output = Linear('Discriminator.Output', 128, 1, output)
        # output = Linear('Discriminator.Output', 512, 1, output)

        return tf.reshape(output, [-1]), tf.reshape(cls_output, [-1, opt.N_CLS]), tf.reshape(de_output,
                                                                                             [-1, opt.LATENT_DIM])

else:
    '''global algorithm: D.'''
    def discriminator(x, z, k):
        '''Four Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11, 13.'''
        if opt.WINDOW_SIZE == 7:
            output = tf.reshape(x, [-1, opt.CHANNEL, 7, 7])

            output = Conv2D('Discriminator.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='SAME', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

        elif opt.WINDOW_SIZE == 9:
            output = tf.reshape(x, [-1, opt.CHANNEL, 9, 9])

            output = Conv2D('Discriminator.1', opt.CHANNEL, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.2', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.3', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

        elif opt.WINDOW_SIZE == 11:
            output = tf.reshape(x, [-1, opt.CHANNEL, 11, 11])

            output = Conv2D('Discriminator.1', opt.CHANNEL, opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.2', opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.3', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.4', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

        elif opt.WINDOW_SIZE == 13:
            output = tf.reshape(x, [-1, opt.CHANNEL, 13, 13])

            output = Conv2D('Discriminator.1', opt.CHANNEL, opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.2', opt.DIM, 2 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.3', 2 * opt.DIM, 4 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)

            output = Conv2D('Discriminator.4', 4 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)
            output = Conv2D('Discriminator.5', 8 * opt.DIM, 8 * opt.DIM, 3, output, padding='VALID', stride=1)
            output = leakyrelu(output)
            output = tf.layers.dropout(output, rate=1. - keep_prob)
        else:
            raise NotImplementedError

        output = tf.reshape(output, [-1, 3 * 3 * 8 * opt.DIM])
        '''treat D as an extractor, this is the output'''
        de_output = Linear('Discriminator.de1', 3 * 3 * 8 * opt.DIM, opt.LATENT_DIM, output)

        # """Scheme 1."""
        # '''for classification'''
        # cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, opt.N_CLS, output)
        # # cls_output = Linear('Discriminator.c1', 3 * 3 * 4 * opt.DIM, 256, output)
        # # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
        # # cls_output = tf.nn.softmax(cls_output)
        #
        # zk_output = tf.concat([z, k], 1)
        # zk_output = Linear('Discriminator.zk1', opt.LATENT_DIM + opt.N_GM, 512, zk_output)
        # # zk_output = Linear('Discriminator.zk1', opt.LATENT_DIM + opt.N_GM, 512, zk_output)
        # zk_output = leakyrelu(zk_output)
        # # zk_output = tf.layers.dropout(zk_output, rate=.2)
        # zk_output = tf.layers.dropout(zk_output, rate=1. - keep_prob)
        #
        # output = tf.concat([output, zk_output], 1)
        # output = Linear('Discriminator.zxk1', 3 * 3 * 8 * opt.DIM + 512, 128, output)
        # # output = Linear('Discriminator.zxk1', 3 * 3 * 4 * opt.DIM + 512, 512, output)
        # output = leakyrelu(output)
        # # output = tf.layers.dropout(output, rate=.2)
        # output = tf.layers.dropout(output, rate=1. - keep_prob)
        #
        # output = Linear('Discriminator.Output', 128, 1, output)
        # # output = Linear('Discriminator.Output', 512, 1, output)

        '''Scheme 2.'''
        '''for classification'''
        # cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, opt.N_CLS, output)
        # # cls_output = Linear('Discriminator.c1', 3 * 3 * 4 * opt.DIM, 256, output)
        # # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
        # # cls_output = tf.nn.softmax(cls_output)

        zk_output = tf.concat([z, k], 1)
        zk_output = Linear('Discriminator.zk1', opt.LATENT_DIM + opt.N_GM, 512, zk_output)
        zk_output = leakyrelu(zk_output)
        zk_output = tf.layers.dropout(zk_output, rate=1. - keep_prob)

        output = tf.concat([output, zk_output], 1)
        output = Linear('Discriminator.zxk1', 3 * 3 * 8 * opt.DIM + 512, 128, output)
        # output = Linear('Discriminator.zxk1', 3 * 3 * 4 * opt.DIM + 512, 512, output)
        output = leakyrelu(output)
        # output = tf.layers.dropout(output, rate=.2)
        output = tf.layers.dropout(output, rate=1. - keep_prob)

        cls_output = Linear('Discriminator.c1', 128, opt.N_CLS, output)
        # cls_output = Linear('Discriminator.c1', 3 * 3 * 4 * opt.DIM, 256, output)
        # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
        # cls_output = tf.nn.softmax(cls_output)

        output = Linear('Discriminator.Output', 128, 1, output)
        # output = Linear('Discriminator.Output', 512, 1, output)


        return tf.reshape(output, [-1]), tf.reshape(cls_output, [-1, opt.N_CLS]), tf.reshape(de_output,
                                                                                           [-1, opt.LATENT_DIM])

"""
losses
"""
real_x_int = tf.compat.v1.placeholder(tf.int32, shape=[opt.BATCH_SIZE, opt.OUTPUT_DIM])
real_y_int = tf.compat.v1.placeholder(tf.int64, shape=[opt.BATCH_SIZE])
keep_prob = tf.compat.v1.placeholder(tf.float32)
real_x = tf.reshape(2 * ((tf.cast(real_x_int, tf.float32) / 256.) - .5), [opt.BATCH_SIZE, opt.OUTPUT_DIM])
real_x += tf.random.uniform(shape=[opt.BATCH_SIZE, opt.OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize

q_z = extractor(real_x)
q_k_logits, q_k = hyper_extractor(q_z)
q_k_probs = tf.nn.softmax(q_k_logits)
rec_x = generator(q_z, real_y_int)

hyper_p_z = tf.random.normal([opt.BATCH_SIZE, opt.LATENT_DIM])
hyper_p_k = tf.one_hot(indices=prior_k.sample(opt.BATCH_SIZE), depth=opt.N_GM)
p_z = hyper_generator(hyper_p_k, hyper_p_z)
fake_y = np.random.randint(0, opt.N_CLS, size=opt.BATCH_SIZE)
fake_y_tensor = tf.convert_to_tensor(np.random.randint(0, opt.N_CLS, size=opt.BATCH_SIZE))
fake_x = generator(p_z, fake_y_tensor)
# stop gradient for correlation loss
fake_x_ng = tf.stop_gradient(fake_x)

if opt.MODE is 'local':
    dscore_real, clogits_real, _ = discriminator(real_x, q_z)
    dscore_fake, clogits_fake, de_z = discriminator(fake_x, p_z)
    disc_fake, disc_real = [], []
    disc_fake.append(hyper_discriminator(p_z, hyper_p_k))
    disc_real.append(hyper_discriminator(q_z, q_k))
    disc_fake.append(dscore_fake)
    disc_real.append(dscore_real)
else:
    '''global'''
    disc_real, clogits_real, _ = discriminator(real_x, q_z, q_k)
    disc_fake, clogits_fake, de_z = discriminator(fake_x, p_z, hyper_p_k)

# stop gradient for correlation loss
fake_de_z = tf.stop_gradient(de_z)

gen_params = lib.params_with_name('Generator')
ext_params = lib.params_with_name('Extractor')
disc_params = lib.params_with_name('Discriminator')

'''reconstruction loss'''
rec_loss = 1.*distance(real_x, rec_x, opt.DISTANCE_TYPE) if opt.with_REC is True else None
'''adversarial loss'''

'''classification loss'''
cls_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=clogits_real,
                                                                          labels=tf.one_hot(indices=real_y_int,
                                                                                            depth=opt.N_CLS)))


cls_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=clogits_fake,
                                                                          labels=tf.one_hot(indices=fake_y,
                                                                                            depth=opt.N_CLS)))

'''correlation loss'''
corr_loss = tf.reduce_mean(correlation(p_z, de_z)) if opt.with_CORR is True else None

gen_loss, disc_loss, gen_train_op, disc_train_op = overall_loss(disc_fake, disc_real,
                                                                gen_params + ext_params, disc_params,
                                                                lr=opt.LR, beta1=opt.BETA1, beta2=opt.BETA2,
                                                                s_f=None,
                                                                rec_loss=rec_loss,
                                                                cls_real_loss=cls_real_loss,
                                                                cls_fake_loss=cls_fake_loss,
                                                                corr_loss=corr_loss,
                                                                adam=True)

correct_pred_real = tf.equal(tf.argmax(clogits_real, axis=1), real_y_int)
# correct_pred_fake = tf.equal(tf.argmax(clogits_fake, axis=1), fake_y)
acc_real = tf.reduce_mean(tf.cast(correct_pred_real, tf.float32))
# acc_fake = tf.reduce_mean(tf.cast(correct_pred_fake, tf.float32))

"""
dataset iterator
"""
train_Xy, train_niter_epoch, test_Xy = load(opt.BATCH_SIZE, dataset=opt.DATASET, data_dir=opt.DATA_DIR,
                                            test_ratio=opt.TEST_RATIO,
                                            window_size=opt.WINDOW_SIZE, use_pca=opt.use_PCA)
'''
Define a python generator, easily callable, able to yield images iter by iter.
'''
def inf_train_xy():
    while True:
        for images, labels in train_Xy():
            yield images, labels


saver = tf.compat.v1.train.Saver()

"""
Train loop
"""
run_config = tf.compat.v1.ConfigProto()
run_config.gpu_options.allow_growth = True

epoch_OA = []
epoch_Each = []
epoch_AA = []
epoch_kappa = []
with tf.compat.v1.Session(config=run_config) as session:
    session.run(tf.compat.v1.global_variables_initializer())
    gen = inf_train_xy()

    total_num = np.sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
    print("\nTotol number of parameters:", int(total_num))

    for iter in xrange(opt.ITERS):
        '''start training G at iteration 1'''
        if iter > 0:
            _data, _label = gen.next()
            _gen_loss, _ = session.run([gen_loss, gen_train_op],
                                       feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: (1. - opt.DR_RATE)}
                                       )

        for critic in xrange(opt.CRITIC_ITERS):
            _data, _label = gen.next()
            _disc_loss, _ = session.run(
                [disc_loss, disc_train_op],
                feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: (1. - opt.DR_RATE)}
            )
        if iter % train_niter_epoch == 0:
            print("\tepoch:{}, gen_loss:{}, disc_loss:{}, cls_real_loss:{}, rec_loss: None, corr_loss: None".format(
                int(iter / (train_niter_epoch)),
                session.run(gen_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
                session.run(disc_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
                session.run(cls_real_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.})
            ))
            # print("\tepoch:{}, gen_loss:{}, disc_loss:{}, cls_real_loss:{}, rec_loss: {}, corr_loss: {}".format(
            #     int(iter / (train_niter_epoch)),
            #     session.run(gen_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
            #     session.run(disc_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
            #     session.run(cls_real_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
            #     session.run(rec_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
            #     session.run(corr_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.})))
            train_acc_real = acc_real.eval(feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.})
            print("\ttrain_acc_real:{}".format(100 * train_acc_real))
            '''
            testing
            '''
            # test only if train_acc_real is larger than opt.TEST_TH.
            if train_acc_real < opt.TEST_TH:
                epoch_OA.append(0)
                epoch_Each.append([0] * opt.N_CLS)
                epoch_AA.append(0)
                epoch_kappa.append(0)
            else:
                labels = np.array([], dtype=np.int64)
                preds = np.array([], dtype=np.int64)
                for img, label in test_Xy():
                    pred = np.argmax(session.run(clogits_real, feed_dict={real_x_int: img, real_y_int: label}),
                                     axis=1)
                    labels = np.append(labels, label)
                    preds = np.append(preds, pred)

                OA = accuracy_score(y_true=labels, y_pred=preds)
                confusion = confusion_matrix(y_true=labels, y_pred=preds)
                each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
                AA = np.mean(each_acc)
                kappa = cohen_kappa_score(y1=labels, y2=preds)

                epoch_OA.append(OA)
                epoch_Each.append(each_acc)
                epoch_AA.append(AA)
                epoch_kappa.append(kappa)

                '''Trace the best testing results of one training time. Consider best OA firstly.'''
                '''Tracing each best is wrong!'''
                best_epoch_OA = np.max(epoch_OA)
                best_epoch_OA_index = np.argmax(epoch_OA)

                epoch_Each_ = np.reshape(epoch_Each, (-1, opt.N_CLS))
                best_OA_epoch_Each_ = epoch_Each_[int(best_epoch_OA_index), :]

                best_OA_epoch_AA = epoch_AA[int(best_epoch_OA_index)]

                best_OA_epoch_kappa = epoch_kappa[int(best_epoch_OA_index)]

                print("\tOA:{}, AA:{}, kappa:{}; best OA: {} at epoch {}, AA this time:{}, kappa this time:{}."
                      .format(100 * OA, 100 * AA, 100 * kappa,
                              100 * best_epoch_OA, best_epoch_OA_index, 100 * best_OA_epoch_AA,
                              100 * best_OA_epoch_kappa))

                '''Save model: save the best only.'''
                if OA == best_epoch_OA:
                    if not os.listdir(os.path.join(outf, 'ckpt')):
                        pass
                    else:
                        to_delete = os.listdir(os.path.join(outf, 'ckpt'))
                        for file in to_delete:
                            os.remove(os.path.join(os.path.join(outf, 'ckpt'), file))
                    save_path = saver.save(session, os.path.join(os.path.join(outf, 'ckpt'), 'epoch{}_model.ckpt'.
                                                                 format(int(iter / train_niter_epoch))
                                                                 ))
                    '''Retest to get confusion array. This is not needed.'''
                    # for img, label in test_Xy():
                    #     pred = np.argmax(session.run(clogits_real, feed_dict={real_x_int: img, real_y_int: label}),
                    #                      axis=1)
                    #     labels = np.append(labels, label)
                    #     preds = np.append(preds, pred)
                    # confusion = confusion_matrix(y_true=labels, y_pred=preds)

                    '''Record results for each training time.'''
                    with open(logfile, 'w') as f:
                        f.write('\n')
                        f.write('DATASET: {}'.format(str(opt.DATASET)))
                        f.write('\n')
                        f.write(
                            'WINDOW_SIZE: {}, CRITIC_ITERS: {}, N_GM:{}, EMBEDDING_DIM: {}, BN: {}'.format(
                                str(opt.WINDOW_SIZE),
                                str(opt.CRITIC_ITERS),
                                str(opt.N_GM),
                                str(opt.EMBEDDING_TYPE),
                                str(opt.BN_FLAG)
                            ))
                        f.write('\n')
                        f.write('best OA at this training time: {}'.format(100 * best_epoch_OA))
                        f.write('\n')
                        f.write('Each: {}'.format(100 * best_OA_epoch_Each_))
                        f.write('\n')
                        f.write('AA: {}'.format(100 * best_OA_epoch_AA))
                        f.write('\n')
                        f.write('kappa: {}'.format(100 * best_OA_epoch_kappa))
                        f.write('\n')
                        f.write('{}'.format(str(confusion)))
                        f.write('\n')



"""
Build the Classification Maps. 
Uncomment the below session.
Comment the above session to avoid retraining.
"""
# '''load the original data.'''
# raw_X, raw_y, map_Xy, niter_epoch = load_map(opt.BATCH_SIZE, opt.DATASET, opt.DATA_DIR, opt.WINDOW_SIZE)
# def inf_map_xy():
#     while True:
#         for images, labels in map_Xy():
#             yield images, labels
# 
# with tf.compat.v1.Session(config=run_config) as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     saver.restore(sess, 'Path_2_Saved/*.ckpt')
#     preds_map = np.array([], dtype=np.int64)
#     for img, label in map_Xy():
#         pred_map = np.argmax(sess.run(clogits_real, feed_dict={real_x_int: img, real_y_int: label}),
#                          axis=1)
# 
#         labels_map = np.append(labels_map, label)
#         preds_map = np.append(preds_map, pred_map)
#     labels_map = labels_map + 1
#     preds_map = preds_map + 1
#     for idx in range(len(labels_map)):
#         if labels_map[idx] == 0:
#             preds_map[idx] = 0
# 
#     if opt.DATASET == 'Salinas':
#         preds_map = np.reshape(preds_map, (512, 217))
#     elif opt.DATASET == 'PaviaU':
#         preds_map = preds_map[0:-24]
#         preds_map = np.reshape(preds_map, (610, 340))
#     elif opt.DATASET == 'Indian':
#         preds_map = preds_map[0:-31]
#         preds_map = np.reshape(preds_map, (145, 145))
# 
#     spectral.save_rgb(str(opt.DATASET)+"predictions.jpg", preds_map.astype(int), colors=spectral.spy_colors)
#     spectral.save_rgb(str(opt.DATASET) + "groundtruth.jpg", raw_y, colors=spectral.spy_colors)
