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
outf += '.MODE_GAC-'
outf += opt.MODE_GAC
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
def leakyrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


'''
Loss function.
'''
def gan(disc_fake, disc_real, gen_params, disc_params, lr=2e-4, beta1=0.5, beta2=0.999, s_f = None,
         cls_real_loss=None, cls_fake_loss=None, corr_loss=None, adam=False):
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.ones_like(disc_fake)
    ))
    gen_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.zeros_like(disc_real)
    ))

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros_like(disc_fake)
    ))
    disc_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones_like(disc_real)
    ))

    if s_f is not None:
        gen_loss += s_f

    if (cls_real_loss is not None) and (cls_fake_loss is not None):
        gen_loss += cls_fake_loss
        disc_loss += cls_real_loss + cls_fake_loss

    if corr_loss is not None:
        gen_loss -= 0.5 * corr_loss
        disc_loss -= 0.5 * corr_loss

    '''Use Adam optimizer'''
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
        '''Use RAdam optimizer'''
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


"""
We have tried to use wgan ans wgan_gp, but seen apparent performance decline."""
def wali(disc_fake, disc_real, gen_params, disc_params, lr=5e-5,
         rec_loss=None, cls_real_loss=None, cls_fake_loss=None, corr_loss=None, adam=False):
    gen_loss = -tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)


    if (cls_real_loss is not None) and (cls_fake_loss is not None):
        gen_loss += cls_real_loss + cls_fake_loss
        disc_loss += cls_real_loss + cls_fake_loss

    if corr_loss is not None:
        gen_loss -= 0.5 * corr_loss
        disc_loss -= 0.5 * corr_loss

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=lr
    ).minimize(gen_loss, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=lr
    ).minimize(disc_loss, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

    return gen_loss, disc_loss, clip_disc_weights, gen_train_op, disc_train_op, clip_ops


def wgan_gp(disc_fake, disc_real, gradient_penalty, gen_params, disc_params, lr=1e-4, beta1=0.5, beta2=0.999,
            cls_real_loss=None, cls_fake_loss=None, corr_loss=None, adam=False):
    gen_loss = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
    disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    disc_loss += gradient_penalty

    if (cls_real_loss is not None) and (cls_fake_loss is not None):
        gen_loss += cls_real_loss + cls_fake_loss
        disc_loss += cls_real_loss + cls_fake_loss

    if corr_loss is not None:
        gen_loss -= 0.5 * corr_loss
        disc_loss -= 0.5 * corr_loss

    '''use Adam optimizer.'''
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
        '''use RAdam optimizer.'''
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
generator: G.
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

    '''Three Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11.'''
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
    else:
        raise NotImplementedError

    return tf.reshape(output, [-1, opt.OUTPUT_DIM])


'''
discriminator: D.
'''
def discriminator(x):
    '''Three Spatial Neighbourhood Sizes (WINDOW_SIZE): 7, 9, 11.'''
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

    '''for classification'''
    cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, opt.N_CLS, output)
    # cls_output = Linear('Discriminator.c1', 3 * 3 * 8 * opt.DIM, 256, output)
    # cls_output = Linear('Discriminator.c2', 256, opt.N_CLS, output)
    # cls_output = tf.nn.softmax(cls_output)

    output = Linear('Discriminator.5', 3 * 3 * 8 * opt.DIM, 128, output)
    # output = Linear('Discriminator.zx1', 3 * 3 * 8 * opt.DIM + 512, 512, output)
    output = leakyrelu(output)

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


p_z = tf.random.normal([opt.BATCH_SIZE, opt.LATENT_DIM])
fake_y = np.random.randint(0, opt.N_CLS, size=opt.BATCH_SIZE)
fake_y_tensor = tf.convert_to_tensor(np.random.randint(0, opt.N_CLS, size=opt.BATCH_SIZE))
fake_x = generator(p_z, fake_y_tensor)

disc_real, clogits_real, _ = discriminator(real_x)
disc_fake, clogits_fake, de_z = discriminator(fake_x)


gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

'''adversarial loss'''

'''classification loss'''
cls_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=clogits_real,
                                                                       labels=tf.one_hot(indices=real_y_int,
                                                                                         depth=opt.N_CLS)))


cls_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=clogits_fake,
                                                                       labels=tf.one_hot(indices=fake_y,
                                                                                         depth=opt.N_CLS)))

'''correlation loss'''
corr_loss = None

if opt.MODE_GAC == 'gan':
    gen_loss, disc_loss, gen_train_op, disc_train_op = gan(disc_fake, disc_real,
                                                            gen_params, disc_params,
                                                            lr=opt.LR, beta1=opt.BETA1, beta2=opt.BETA2,
                                                            s_f=None,
                                                            cls_real_loss=cls_real_loss, cls_fake_loss=cls_fake_loss,
                                                            corr_loss=None,
                                                            adam=True)
elif opt.MODE_GAC == 'wgan':
    gen_loss, disc_loss, clip_disc_weights, gen_train_op, disc_train_op, clip_ops = wgan(disc_fake, disc_real,
                                                            gen_params, disc_params, lr=5e-5,
                                                            cls_real_loss=cls_real_loss, cls_fake_loss=cls_fake_loss,
                                                            corr_loss=None)
elif opt.MODE_GAC == 'wgan_gp':
    alpha = tf.random.uniform(
        shape=[opt.BATCH_SIZE, 1, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_x - real_x
    interpolates = real_x + (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 10. * (tf.reduce_mean((slopes - 1.) ** 2))

    gen_loss, disc_loss, gen_train_op, disc_train_op = wgan_gp(disc_fake, disc_real, gradient_penalty,
                                                            gen_params, disc_params,
                                                            lr=opt.LR, beta1=opt.BETA1, beta2=opt.BETA2,
                                                            cls_real_loss=cls_real_loss, cls_fake_loss=cls_fake_loss,
                                                            corr_loss=None,
                                                            adam=True)
else:
    raise NotImplementedError

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
            print("\tepoch:{}, gen_loss:{}, disc_loss:{}, cls_real_loss:{}".format(
                int(iter / (train_niter_epoch)),
                session.run(gen_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
                session.run(disc_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.}),
                session.run(cls_real_loss, feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.})
            ))
            train_acc_real = acc_real.eval(feed_dict={real_x_int: _data, real_y_int: _label, keep_prob: 1.})
            print("\ttrain_acc_real:{}".format(100 * train_acc_real))
        


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
