import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import identity, softmax
import numpy as np
import theano
import theano.tensor as T
from parmesan.layers import (ListIndexLayer, NormalizeLayer,
                             ScaleAndShiftLayer, DecoderNormalizeLayer,
                             DenoiseLayer,)
import os
import uuid
import parmesan
import os
import gzip
import cPickle
import numpy as np
import tarfile
import fnmatch
import scipy


class ContrastNorm(object):
    def __init__(self, scale=55, epsilon=1e-8):
        self.scale = np.float32(scale)
        self.epsilon = np.float32(epsilon)

    def apply(self, data, copy=False):
        if copy:
            data = np.copy(data)
        data_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], np.product(data.shape[1:]))

        assert len(data.shape) == 2, 'Contrast norm on flattened data'

        data -= data.mean(axis=1)[:, np.newaxis]

        norms = np.sqrt(np.sum(data ** 2, axis=1)) / self.scale
        norms[norms < self.epsilon] = np.float32(1.)

        data /= norms[:, np.newaxis]

        if data_shape != data.shape:
            data = data.reshape(data_shape)

        return data

class ZCA(object):
    """
    Code copied from https://github.com/arasmus/ladder/blob/master/nn.py
    """
    def __init__(self, n_components=None, data=None, filter_bias=0.1):
        self.filter_bias = np.float32(filter_bias)
        self.P = None
        self.P_inv = None
        self.n_components = 0
        self.is_fit = False
        if n_components and data:
            self.fit(n_components, data)

    def fit(self, n_components, data):
        if len(data.shape) == 2:
            self.reshape = None
        else:
            assert n_components == np.product(data.shape[1:]), \
                'ZCA whitening components should be %d for convolutional data'\
                % np.product(data.shape[1:])
            self.reshape = data.shape[1:]

        data = self._flatten_data(data)
        assert len(data.shape) == 2
        n, m = data.shape
        self.mean = np.mean(data, axis=0)

        bias = self.filter_bias * scipy.sparse.identity(m, 'float32')
        cov = np.cov(data, rowvar=0, bias=1) + bias
        eigs, eigv = scipy.linalg.eigh(cov)

        assert not np.isnan(eigs).any()
        assert not np.isnan(eigv).any()
        assert eigs.min() > 0

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        sqrt_eigs = np.sqrt(eigs)
        self.P = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
        assert not np.isnan(self.P).any()
        self.P_inv = np.dot(eigv * sqrt_eigs, eigv.T)

        self.P = np.float32(self.P)
        self.P_inv = np.float32(self.P_inv)

        self.is_fit = True

    def apply(self, data, remove_mean=True):
        data = self._flatten_data(data)
        d = data - self.mean if remove_mean else data
        return self._reshape_data(np.dot(d, self.P))

    def inv(self, data, add_mean=True):
        d = np.dot(self._flatten_data(data), self.P_inv)
        d += self.mean if add_mean else 0.
        return self._reshape_data(d)

    def _flatten_data(self, data):
        if self.reshape is None:
            return data
        assert data.shape[1:] == self.reshape
        return data.reshape(data.shape[0], np.product(data.shape[1:]))

    def _reshape_data(self, data):
        assert len(data.shape) == 2
        if self.reshape is None:
            return data
        return np.reshape(data, (data.shape[0],) + self.reshape)


class Depooling(lasagne.layers.Layer):
    """
    code from
    https://gist.github.com/kastnerkyle/f3f67424adda343fef40

    """
    def __init__(self, incoming, factor, ignore_border=True, **kwargs):
        super(Depooling, self).__init__(incoming, **kwargs)
        self.factor = factor
        self.ignore_border = ignore_border

    def get_output_for(self, input, **kwargs):
        stride = input.shape[2]
        offset = input.shape[3]
        in_dim = stride * offset
        out_dim = in_dim * self.factor * self.factor

        upsamp_matrix = T.zeros((in_dim, out_dim))
        rows = T.arange(in_dim)
        cols = rows*self.factor + (rows/stride * self.factor * offset)
        upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
        flat = T.reshape(
            input,
            (input.shape[0], self.output_shape[1],
             input.shape[2] * input.shape[3]))

        up_flat = T.dot(flat, upsamp_matrix)
        upsamp = T.reshape(up_flat, (input.shape[0], self.output_shape[1],
                                     self.output_shape[2], self.output_shape[3]))

        return upsamp

    def get_output_shape_for(self, input_shape):
        shp = list(input_shape)
        shp = shp[:2] + [shp[2]*self.factor, shp[3]*self.factor]
        return tuple(shp)


class MyInit(lasagne.init.Initializer):
    """Sample initial weights from the Gaussian distribution.
    Initial weight parameters are sampled from N(mean, std).
    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=1.0, mean=0.0):
        self.std = std
        self.mean = mean

    # std one should reproduce rasmus init...
    def sample(self, shape):
        return lasagne.utils.floatX(lasagne.random.get_rng().normal(
            self.mean, self.std, size=shape) /
                      np.sqrt(shape[0]))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-lambdas", type=str,
                    default='0,0,0,0,0,0,4')
parser.add_argument("-lr", type=str, default='0.0005')
parser.add_argument("-optimizer", type=str, default='adam')
parser.add_argument("-init", type=str, default='None')
parser.add_argument("-initval", type=str, default='relu')
parser.add_argument("-gradclip", type=str, default='1')
args = parser.parse_args()


num_classes = 10
batch_size = 100  # fails if batch_size != batch_size
num_labels = 100

num_labeled = 4000

output_folder = "logs/mnist_ladder_conv" + str(uuid.uuid4())[:18].replace('-', '_')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = os.path.join(output_folder, 'results.log')

with open(output_file, 'wb') as f:
    f.write("#"*80 + "\n")
    for name, val in sorted(vars(args).items()):
        s = str(name) + " "*(40-len(name)) + str(val)
        f.write(s + "\n")
    f.write("#"*80 + "\n")

optimizers = {'adam': lasagne.updates.adam,
              'adadelta': lasagne.updates.adadelta,
              'rmsprop': lasagne.updates.rmsprop,
              'sgd': lasagne.updates.sgd,
              'nag': lasagne.updates.nesterov_momentum
              }
optimizer = optimizers[args.optimizer]

if args.init == 'None':  # default to antti rasmus init
    init = MyInit()
else:
    initval = float(args.initval)
    inits = {'he': lasagne.init.HeUniform(initval),
             'glorot': lasagne.init.HeUniform(initval),
             'uniform': lasagne.init.HeUniform(initval),
             'normal': lasagne.init.HeUniform(initval)}
    init = inits[args.init]


if args.gradclip == 'None':
    gradclip = None
else:
    gradclip = float(args.gradclip)

unit = lasagne.nonlinearities.leaky_rectify
lasagne.random.set_rng(np.random.RandomState(seed=1))

[x_train, targets_train, x_valid,
 targets_valid, x_test, targets_test] = parmesan.datasets.cifar10()

x_train = x_train.astype('float32').reshape((-1, 32*32*3))
x_valid = x_valid.astype('float32').reshape((-1, 32*32*3))
x_test = x_test.astype('float32').reshape((-1, 32*32*3))
targets_train = targets_train.astype('int32')
targets_valid = targets_valid.astype('int32')
targets_test = targets_test.astype('int32')


contranst_norm = ContrastNorm()
zca = ZCA()
zca.fit(3072, x_train)
assert zca.is_fit
x_train = zca.apply(contranst_norm.apply(x_train))
x_valid = zca.apply(contranst_norm.apply(x_valid))
x_test = zca.apply(contranst_norm.apply(x_test))

np.random.seed(1)
shuffle = np.random.permutation(x_train.shape[0])
x_train_lab = x_train[:num_labeled]
targets_train_lab = targets_train[:num_labeled]
labeled_slice = slice(0, num_labels)
unlabeled_slice = slice(num_labels, 2*num_labels)

lambdas = map(float, args.lambdas.split(','))

assert len(lambdas) == 7
print "Lambdas: ", lambdas


num_classes = 10
lr = float(args.lr)
h,w, c = 32, 32, 3
noise = 0.3
num_epochs = 300
start_decay = 50
sym_x = T.matrix('sym_x')
sym_t = T.ivector('sym_t')
sh_lr = theano.shared(lasagne.utils.floatX(lr))

z_pre0 = InputLayer(shape=(None, x_train.shape[1]))
z_pre0 = ReshapeLayer(z_pre0, ([0], c, h, w))
z0 = z_pre0   # for consistency with other layers
z_noise0 = GaussianNoiseLayer(z0, sigma=noise, name='enc_noise0')
h0 = z_noise0  # no nonlinearity on input


def get_unlab(l):
    return SliceLayer(l, indices=slice(num_labels, None), axis=0)


def create_encoder_conv(incoming, num_filters, nonlinearity, layer_num, maxpool=False):
    i = layer_num
    if maxpool:
        incoming = MaxPool2DLayer(incoming, pool_size=2)
    z_pre = Conv2DLayer(
        incoming=incoming, nonlinearity=identity, b=None,
        name='enc_dense%i' % i, W=init, num_filters=num_filters, filter_size=3, pad=1)
    norm_list = NormalizeLayer(
        z_pre, return_stats=True, name='enc_normalize%i' % i,
        stat_indices=unlabeled_slice)
    z = ListIndexLayer(norm_list, index=0, name='enc_index%i' % i)
    z_noise = GaussianNoiseLayer(z, sigma=noise, name='enc_noise%i' % i)
    h = NonlinearityLayer(
        ScaleAndShiftLayer(z_noise, name='enc_scale%i' % i),
        nonlinearity=nonlinearity, name='enc_nonlin%i' % i)
    return h, z, z_noise, norm_list


def create_encoder_dense(incoming, num_units, nonlinearity, layer_num):
    i = layer_num
    z_pre = DenseLayer(
        incoming=incoming, num_units=num_units, nonlinearity=identity, b=None,
        name='enc_dense%i' % i, W=init)
    norm_list = NormalizeLayer(
        z_pre, return_stats=True, name='enc_normalize%i' % i,
        stat_indices=unlabeled_slice)
    z = ListIndexLayer(norm_list, index=0, name='enc_index%i' % i)
    z_noise = GaussianNoiseLayer(z, sigma=noise, name='enc_noise%i' % i)
    h = NonlinearityLayer(
        ScaleAndShiftLayer(z_noise, name='enc_scale%i' % i),
        nonlinearity=nonlinearity, name='enc_nonlin%i' % i)
    return h, z, z_noise, norm_list


def denoise_conv(u, z, name):
    num_filters, height, width = z.output_shape[1:]

    z_shp = ([0],) + z.output_shape[1:]

    denoise = DenoiseLayer(z_net=flatten(z), u_net=flatten(u))
    denoise = ReshapeLayer(denoise, z_shp)

    #uz = ConcatLayer([u, z], axis=1, name=name)
    #print z.output_shape, u.output_shape
    #conv_lin = Conv2DLayer(uz, nonlinearity=identity, num_filters=num_filters,
    #                       filter_size=3, pad=1, name=name+ "_lin")
    #conv_nonlin = Conv2DLayer(uz,
    #                          num_filters=num_filters, filter_size=3, pad=1, name=name+"_nonlin")
    #out_conv = ConcatLayer([conv_lin, conv_nonlin], axis=1, name=name)

    #denoise = ElemwiseSumLayer([conv_lin, conv_nonlin], name=name+"_sum")
    #denoise = Conv2DLayer(out_conv, nonlinearity=T.nnet.sigmoid, num_filters=num_filters,
    #                       filter_size=3, pad=1, name=name+ "_lin")
    return denoise


def create_decoder_dense(z_hat_in, z_noise, norm_list, layer_num):
    i = layer_num
    num_units = np.prod(z_noise.output_shape[1:])
    dense = DenseLayer(z_hat_in, num_units=num_units, name='dec_dense%i' % i,
                       W=init, nonlinearity=identity)
    num_filters, height, width = z_noise.output_shape[1:]
    conv = ReshapeLayer(dense, (-1, num_filters, height, width))

    normalize = NormalizeLayer(conv, name='dec_normalize%i' % i)
    u = ScaleAndShiftLayer(normalize, name='dec_scale%i' % i)
    lateral = get_unlab(z_noise)

    z_hat = denoise_conv(u, lateral, name='dec_denoise%i' % i)
    print "z_hat%i:"%i, lasagne.layers.get_output(z_hat, sym_x).eval({sym_x: x_train[:200]}).shape
    mean = ListIndexLayer(norm_list, index=1, name='dec_index_mean%i' % i)
    var = ListIndexLayer(norm_list, index=2, name='dec_index_var%i' % i)
    z_hat_bn = DecoderNormalizeLayer(z_hat, mean=mean, var=var,
                                     name='dec_decnormalize%i' % i)
    return z_hat, z_hat_bn

def create_decoder_conv(z_hat_in, z_noise, norm_list, layer_num):
    i = layer_num
    num_filters, height, width = z_noise.output_shape[1:]
    if z_hat_in.output_shape[2] != z_noise.output_shape[2]:  # if heights are not identical depool
        z_hat_in = Depooling(z_hat_in, factor=2)
        assert z_hat_in.output_shape[2] == z_noise.output_shape[2], "heights are not identical after deppoling, " \
                                                                    "something whent wrong...."

    conv = Conv2DLayer(z_hat_in, num_filters=num_filters,
                        filter_size=3, pad=1, name='dec_dense%i' % i,
                       W=init, nonlinearity=identity)
    normalize = NormalizeLayer(conv, name='dec_normalize%i' % i)
    u = ScaleAndShiftLayer(normalize, name='dec_scale%i' % i)
    lateral = get_unlab(z_noise)

    z_hat = denoise_conv(u, lateral, name='dec_denoise%i' % i)
    print "z_hat%i:"%i, lasagne.layers.get_output(z_hat, sym_x).eval({sym_x: x_train[:200]}).shape
    mean = ListIndexLayer(norm_list, index=1, name='dec_index_mean%i' % i)
    var = ListIndexLayer(norm_list, index=2, name='dec_index_var%i' % i)
    z_hat_bn = DecoderNormalizeLayer(z_hat, mean=mean, var=var,
                                     name='dec_decnormalize%i' % i)
    return z_hat, z_hat_bn



h1, z1, z_noise1, norm_list1 = create_encoder_conv(
    h0, num_filters=64, nonlinearity=unit, layer_num=1, maxpool=False)

h2, z2, z_noise2, norm_list2 = create_encoder_conv(
    h1, num_filters=96, nonlinearity=unit, layer_num=2, maxpool=True)

h3, z3, z_noise3, norm_list3 = create_encoder_conv(
    h2, num_filters=96, nonlinearity=unit, layer_num=3, maxpool=False)

h4, z4, z_noise4, norm_list4 = create_encoder_conv(
    h3, num_filters=96, nonlinearity=unit, layer_num=4, maxpool=True)

h5, z5, z_noise5, norm_list5 = create_encoder_conv(
    h4, num_filters=96, nonlinearity=unit, layer_num=5, maxpool=False)

h6, z6, z_noise6, norm_list6 = create_encoder_dense(
    h4, num_units=10, nonlinearity=softmax, layer_num=6)

l_out_enc = h6

#print "h6:", lasagne.layers.get_output(h6, sym_x).eval({sym_x: x_train[:200]}).shape
h6_dec = get_unlab(l_out_enc)
#print "y_weights_decoder:", lasagne.layers.get_output(h6_dec, sym_x).eval({sym_x: x_train[:200]}).shape

# note that the DenoiseLayer takes a z_indices argument which slices
# the lateral connection from the encoder. For the fully supervised case
# the slice is just all labels.


##### Decoder Layer 6
u6 = ScaleAndShiftLayer(NormalizeLayer(
    h6_dec, name='dec_normalize6'), name='dec_scale6')
z_hat6 = DenoiseLayer(u_net=u6, z_net=get_unlab(z_noise6), name='dec_denoise6')
mean6 = ListIndexLayer(norm_list6, index=1, name='dec_index_mean6')
var6 = ListIndexLayer(norm_list6, index=2, name='dec_index_var6')
z_hat_bn6 = DecoderNormalizeLayer(
    z_hat6, mean=mean6, var=var6, name='dec_decnormalize6')
###########################
#print "z_hat_bn6:", lasagne.layers.get_output(z_hat_bn6, sym_x).eval({sym_x: x_train[:200]}).shape

#z_hat5, z_hat_bn5 = create_decoder_dense(z_hat6, z_noise5, norm_list5, 5)
#z_hat4, z_hat_bn4 = create_decoder_conv(z_hat5, z_noise4, norm_list4, 4)
#z_hat3, z_hat_bn3 = create_decoder_conv(z_hat4, z_noise3, norm_list3, 3)
#z_hat2, z_hat_bn2 = create_decoder_conv(z_hat3, z_noise2, norm_list2, 2)
#z_hat1, z_hat_bn1 = create_decoder_conv(z_hat2, z_noise1, norm_list1, 1)


############################# Decoder Layer 0
# i need this because i also has h0 aka. input layer....
# MAYBE CHANGE THIS TO RECURRENT CONV???
# num_channels_input = z_noise0.output_shape[1]
# conv = Conv2DLayer(z_hat1, num_filters=num_channels_input, name='dec_dense0',
#                     W=init, nonlinearity=identity, filter_size=3, pad=1)
# normalize = NormalizeLayer(conv, name='dec_normalize0')
# u0 = ScaleAndShiftLayer(normalize,  name='dec_scale0')
# lateral0 = get_unlab(z_noise0)
# z_hat0 = denoise_conv(u0, lateral0, name='dec_denoise0')
# z_hat_bn0 = z_hat0   # for consistency
#############################

#print "z_hat_bn0:", lasagne.layers.get_output(
#    z_hat_bn0, sym_x).eval({sym_x: x_train[:200]}).shape

[enc_out_clean, z0_clean, z1_clean, z2_clean,
 z3_clean, z4_clean, z5_clean, z6_clean] = lasagne.layers.get_output(
    [l_out_enc, z0, z1, z2, z3, z4, z5, z6], sym_x, deterministic=True)

# Clean pass of encoder  note that these are both labeled
# and unlabeled so we need to slice
z0_clean = z0_clean[num_labels:]
z1_clean = z1_clean[num_labels:]
z2_clean = z2_clean[num_labels:]
z3_clean = z3_clean[num_labels:]
z4_clean = z4_clean[num_labels:]
z5_clean = z5_clean[num_labels:]
z6_clean = z6_clean[num_labels:]

# noisy pass encoder + decoder
# the output from the decoder is only unlabeled because we slice the top h
[out_enc_noisy,
 #z_hat_bn0_noisy, z_hat_bn1_noisy,
 #z_hat_bn2_noisy, z_hat_bn3_noisy, z_hat_bn4_noisy,
 #z_hat_bn5_noisy,
 z_hat_bn6_noisy] = lasagne.layers.get_output(
    [l_out_enc,
     #z_hat_bn0, z_hat_bn1, z_hat_bn2,
     #z_hat_bn3, z_hat_bn4, z_hat_bn5,
     z_hat_bn6],
     sym_x,  deterministic=False)


# if unsupervised we need ot cut ot the samples with no labels.
out_enc_noisy = out_enc_noisy[:num_labels]

costs = [T.mean(T.nnet.categorical_crossentropy(out_enc_noisy, sym_t))]

# i checkt the blocks code - they do sum over the feature dimension
costs += [lambdas[6]*T.sqr(z6_clean.flatten(2) - z_hat_bn6_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[5]*T.sqr(z5_clean.flatten(2) - z_hat_bn5_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[4]*T.sqr(z4_clean.flatten(2) - z_hat_bn4_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[3]*T.sqr(z3_clean.flatten(2) - z_hat_bn3_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[2]*T.sqr(z2_clean.flatten(2) - z_hat_bn2_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[1]*T.sqr(z1_clean.flatten(2) - z_hat_bn1_noisy.flatten(2)).mean(axis=1).mean()]
#costs += [lambdas[0]*T.sqr(z0_clean.flatten(2) - z_hat_bn0_noisy.flatten(2)).mean(axis=1).mean()]


cost = sum(costs)
# prediction passes

print "cost:", cost.eval({sym_x: x_train[:200], sym_t: targets_train[:200]})

collect_out = lasagne.layers.get_output(
    l_out_enc, sym_x, deterministic=True, collect=True)


# Get list of all trainable parameters in the network.
all_params = lasagne.layers.get_all_params(z_hat_bn6, trainable=True)
print ""*20 + "PARAMETERS" + "-"*20
for p in all_params:
    print p.name, p.get_value().shape
print "-"*60

if gradclip is not None:
    all_grads = [T.clip(g, -gradclip, gradclip)
                 for g in T.grad(cost, all_params)]
else:
    all_grads = T.grad(cost, all_params)

updates = optimizer(all_grads, all_params, learning_rate=sh_lr)

f_clean = theano.function([sym_x], enc_out_clean)

f_train = theano.function([sym_x, sym_t],
                          [cost, out_enc_noisy] + costs,
                          updates=updates, on_unused_input='warn')

f_collect = theano.function([sym_x],   # NO UPDATES !!!!!!! FOR COLLECT
                            [collect_out], on_unused_input='warn')


num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
loss = []


def train_epoch_semisupervised(x):
    confusion_train = parmesan.utils.ConfusionMatrix(num_classes)
    losses = []
    shuffle = np.random.permutation(x.shape[0])
    x = x[shuffle]
    for i in range(num_batches_train):
        idx = range(i*batch_size, (i+1)*batch_size)
        x_unsup = x[idx]

        # add labels
        idx_lab = np.random.choice(num_labeled, num_labels)
        x_lab = x_train_lab[idx_lab]
        targets_lab = targets_train_lab[idx_lab]
        x_batch = np.concatenate([x_lab, x_unsup], axis=0)

        # nb same targets all the time...
        output = f_train(x_batch, targets_lab)
        batch_loss, net_out = output[0], output[1]
        layer_costs = output[2:]
        # cut out preds with labels
        net_out = net_out[:num_labels]

        preds = np.argmax(net_out, axis=-1)
        confusion_train.batchadd(targets_lab, preds)
        losses += [batch_loss]
    return confusion_train, losses, layer_costs

num_collect = 5000
perm = np.random.permutation(x_train.shape[0])[:num_collect]
def test_epoch(x, y):
    confusion_valid = parmesan.utils.ConfusionMatrix(num_classes)
    _ = f_collect(x_train[perm])
    net_out = f_clean(x)
    preds = np.argmax(net_out, axis=-1)
    confusion_valid.batchadd(y, preds)
    return confusion_valid

with open(output_file, 'a') as f:
    f.write('Starting Training !\n')


for epoch in range(num_epochs):
    confusion_train, losses_train, layer_costs = train_epoch_semisupervised(x_train)
    confusion_valid = test_epoch(x_valid, targets_valid)
    confusion_test = test_epoch(x_test, targets_test)

    if any(np.isnan(losses_train)) or any(np.isinf(losses_train)):
        with open(output_file, 'w') as f:
            f.write('*NAN')
        break

    train_acc_cur = confusion_train.accuracy()
    valid_acc_cur = confusion_valid.accuracy()
    test_acc_cur = confusion_test.accuracy()

    if epoch > 3 and train_acc_cur < 0.1:
        with open(output_file, 'a') as f:
            f.write('*No progress')
        break

    if epoch > 30 and train_acc_cur < 0.5:
        with open(output_file, 'a') as f:
            f.write('*slow progres')
        break

    if epoch > start_decay:
        old_lr = sh_lr.get_value()
        new_lr = old_lr - (lr/(num_epochs-start_decay))
        sh_lr.set_value(lasagne.utils.floatX(new_lr))

    str_costs = "\t{}"*len(layer_costs)
    s = ("*EPOCH\t{}\t{}\t{}\t{}\t{}\t{}"+str_costs).format(
        epoch, np.mean(losses_train), train_acc_cur, valid_acc_cur,
        test_acc_cur, sh_lr.get_value(), *layer_costs)
    print s
    with open(output_file, 'a') as f:
        f.write(s + "\n")

