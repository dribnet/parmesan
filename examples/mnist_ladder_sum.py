import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import identity, softmax
import numpy as np
import theano
import theano.tensor as T
from parmesan.layers import (ListIndexLayer, NormalizeLayer,
                             ScaleAndShiftLayer, DecoderNormalizeLayer,
                             DenoiseLayer)
import os
import uuid
import parmesan

class RepeatClassLayer(lasagne.layers.Layer):
    '''
        Takes a input layer of shape (batch_size, n_features) and repeats the
        n_features n times such that the output shape is
        (batch_size, n_repeats, n_featues).

        This can be used for recurrent layers where we want to use the
        same input for all timesteps e.g. in sutskever models.
    '''
    def __init__(self, incomings, n_repeat, **kwargs):
        super(RepeatClassLayer, self).__init__(incomings, **kwargs)

        self.n_repeat = n_repeat

    def get_output_shape_for(self, input_shape):
        #
        if input_shape[0] is None:
            bs = None
        else:
            bs = self.n_repeat*input_shape[0]
        return (bs, input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        assert input.ndim == 2
        bs, n_classes = input.shape
        return T.eye(n_classes).dimshuffle('x', 0, 1).repeat(bs, axis=0).reshape((-1, n_classes))


class RepeatBatchLayer(lasagne.layers.Layer):
    '''
        Takes a input layer of shape (batch_size, n_features) and repeats the
        n_features n times such that the output shape is
        (batch_size, n_repeats, n_featues).

        This can be used for recurrent layers where we want to use the
        same input for all timesteps e.g. in sutskever models.
    '''
    def __init__(self, incomings, n_repeat, **kwargs):
        super(RepeatBatchLayer, self).__init__(incomings, **kwargs)

        self.n_repeat = n_repeat

    def get_output_shape_for(self, input_shape):
        #
        if input_shape[0] is None:
            bs = None
        else:
            bs = self.n_repeat*input_shape[0]
        return (bs, input_shape[1])

    def get_output_for(self, input, *args, **kwargs):
        bs, nin = input.shape
        return input.dimshuffle(0, 'x', 1).repeat(self.n_repeat, axis=1).reshape((-1, nin))


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
                    default='1000,10,0.1,0.1,0.1,0.1,0.1')
parser.add_argument("-lr", type=str, default='0.001')
parser.add_argument("-optimizer", type=str, default='adam')
parser.add_argument("-init", type=str, default='None')
parser.add_argument("-initval", type=str, default='relu')
parser.add_argument("-gradclip", type=str, default='1')
args = parser.parse_args()


num_classes = 10
batch_size = 100  # fails if batch_size != batch_size
num_labels = 100

output_folder = "logs/RNN_SPN_MULTICROP_RAM_SVHN" + str(uuid.uuid4())[:18].replace('-', '_')
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
 targets_valid, x_test, targets_test] = parmesan.datasets.load_mnist_realval()

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
targets_train = targets_train.astype('int32')
targets_valid = targets_valid.astype('int32')
targets_test = targets_test.astype('int32')

np.random.seed(1)
shuffle = np.random.permutation(x_train.shape[0])
x_train_lab = x_train[:num_labels]
targets_train_lab = targets_train[:num_labels]
labeled_slice = slice(0, num_labels)
unlabeled_slice = slice(num_labels, 2*num_labels)

lambdas = map(float, args.lambdas.split(','))

assert len(lambdas) == 7
print "Lambdas: ", lambdas


def create_y(num_classes):
    return T.transpose(
        T.eye(num_classes).repeat(num_classes).reshape((num_classes,-1)))

num_classes = 10
num_inputs = 784
lr = float(args.lr)
noise = 0.3
num_epochs = 300
start_decay = 50
sym_x = T.matrix('sym_x')
sym_t = T.ivector('sym_t')
sh_lr = theano.shared(lasagne.utils.floatX(lr))
y_dec = create_y(num_classes)

z_pre0 = InputLayer(shape=(None, x_train.shape[1]))
z0 = z_pre0   # for consistency with other layers
z_noise0 = GaussianNoiseLayer(z0, sigma=noise, name='enc_noise0')
h0 = z_noise0  # no nonlinearity on input


def get_unlab(l):
    return SliceLayer(l, indices=slice(num_labels, None), axis=0)


def create_encoder(incoming, num_units, nonlinearity, layer_num):
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


def create_decoder(z_hat_in, z_noise, num_units, norm_list, layer_num):
    i = layer_num
    dense = DenseLayer(z_hat_in, num_units=num_units, name='dec_dense%i' % i,
                       W=init, nonlinearity=identity)
    normalize = NormalizeLayer(dense, name='dec_normalize%i' % i)
    u = ScaleAndShiftLayer(normalize, name='dec_scale%i' % i)
    z_hat = DenoiseLayer(u_net=u, z_net=RepeatBatchLayer(get_unlab(z_noise), num_classes), name='dec_denoise%i' % i)
    mean = ListIndexLayer(norm_list, index=1, name='dec_index_mean%i' % i)
    var = ListIndexLayer(norm_list, index=2, name='dec_index_var%i' % i)
    z_hat_bn = DecoderNormalizeLayer(z_hat, mean=mean, var=var,
                                     name='dec_decnormalize%i' % i)
    return z_hat, z_hat_bn


h1, z1, z_noise1, norm_list1 = create_encoder(
    h0, num_units=1000, nonlinearity=unit, layer_num=1)

h2, z2, z_noise2, norm_list2 = create_encoder(
    h1, num_units=500, nonlinearity=unit, layer_num=2)

h3, z3, z_noise3, norm_list3 = create_encoder(
    h2, num_units=250, nonlinearity=unit, layer_num=3)

h4, z4, z_noise4, norm_list4 = create_encoder(
    h3, num_units=250, nonlinearity=unit, layer_num=4)

h5, z5, z_noise5, norm_list5 = create_encoder(
    h4, num_units=250, nonlinearity=unit, layer_num=5)

h6, z6, z_noise6, norm_list6 = create_encoder(
    h4, num_units=10, nonlinearity=softmax, layer_num=6)

l_y = h6

print "h6:", lasagne.layers.get_output(h6, sym_x).eval({sym_x: x_train[:200]}).shape
y_weights_decoder = get_unlab(l_y)
print "y_weights_decoder:", lasagne.layers.get_output(y_weights_decoder, sym_x).eval({sym_x: x_train[:200]}).shape

# note that the DenoiseLayer takes a z_indices argument which slices
# the lateral connection from the encoder. For the fully supervised case
# the slice is just all labels.

# this is a fucked up layer that will introduce zero gradient!!
y_dec = RepeatClassLayer(y_weights_decoder, n_repeat=num_classes)
print "y_dec:", lasagne.layers.get_output(y_dec, sym_x).eval({sym_x: x_train[:200]}).shape
#print "y_dec:", lasagne.layers.get_output(y_dec, sym_x).eval({sym_x: x_train[:200]})

##### Decoder Layer 6
u6 = ScaleAndShiftLayer(NormalizeLayer(
    y_dec, name='dec_normalize6'), name='dec_scale6')
z_hat6 = DenoiseLayer(u_net=u6, z_net=RepeatBatchLayer(get_unlab(z_noise6), num_classes), name='dec_denoise6')
mean6 = ListIndexLayer(norm_list6, index=1, name='dec_index_mean6')
var6 = ListIndexLayer(norm_list6, index=2, name='dec_index_var6')
z_hat_bn6 = DecoderNormalizeLayer(
    z_hat6, mean=mean6, var=var6, name='dec_decnormalize6')
###########################

print "z_hat6:", lasagne.layers.get_output(z_hat6, sym_x).eval({sym_x: x_train[:200]}).shape

z_hat5, z_hat_bn5 = create_decoder(z_hat6, z_noise5, 250, norm_list5, 5)
z_hat4, z_hat_bn4 = create_decoder(z_hat5, z_noise4, 250, norm_list4, 4)
z_hat3, z_hat_bn3 = create_decoder(z_hat4, z_noise3, 250, norm_list3, 3)
z_hat2, z_hat_bn2 = create_decoder(z_hat3, z_noise2, 500, norm_list2, 2)
z_hat1, z_hat_bn1 = create_decoder(z_hat2, z_noise1, 1000, norm_list1, 1)


############################# Decoder Layer 0
# i need this because i also has h0 aka. input layer....
u0 = ScaleAndShiftLayer(  # refactor this...
    NormalizeLayer(
        DenseLayer(z_hat1, num_units=num_inputs, name='dec_dense0', W=init, nonlinearity=identity),
        name='dec_normalize0'), name='dec_scale0')
z_hat0 = DenoiseLayer(u_net=u0, z_net=RepeatBatchLayer(get_unlab(z_noise0), num_classes), name='dec_denoise0')
z_hat_bn0 = z_hat0   # for consistency
#############################

print "z_hat_bn0:", lasagne.layers.get_output(
    z_hat_bn0, sym_x).eval({sym_x: x_train[:200]}).shape

[enc_out_clean, z0_clean, z1_clean, z2_clean,
 z3_clean, z4_clean, z5_clean, z6_clean] = lasagne.layers.get_output(
    [l_y, z0, z1, z2, z3, z4, z5, z6], sym_x, deterministic=True)

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
[y_out_enc_noisy, z_hat_bn0_noisy, z_hat_bn1_noisy,
 z_hat_bn2_noisy, z_hat_bn3_noisy, z_hat_bn4_noisy,
 z_hat_bn5_noisy, z_hat_bn6_noisy] = lasagne.layers.get_output(
    [l_y, z_hat_bn0, z_hat_bn1, z_hat_bn2,
     z_hat_bn3, z_hat_bn4, z_hat_bn5, z_hat_bn6],
     sym_x,  deterministic=False)


# if unsupervised we need ot cut ot the samples with no labels.
y_out_enc_noisy = y_out_enc_noisy[:num_labels]
y_weights_decoder_out_noisy = y_out_enc_noisy[:num_labels]

costs = [T.mean(T.nnet.categorical_crossentropy(y_out_enc_noisy, sym_t))]

# i checkt the blocks code - they do sum over the feature dimension

def ladder_cost(z_clean, z_bn, y):
    z_clean = z_clean.dimshuffle(0, 'x', 1)
    z_bn = z_bn.reshape((num_labels, num_classes, -1))
    x = T.sqr(z_clean - z_bn).mean(axis=2)   # bs x nc
    return T.mean(x*y)


costs += [lambdas[6]*ladder_cost(z6_clean, z_hat_bn6_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[5]*ladder_cost(z5_clean, z_hat_bn5_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[4]*ladder_cost(z4_clean, z_hat_bn4_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[3]*ladder_cost(z3_clean, z_hat_bn3_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[2]*ladder_cost(z2_clean, z_hat_bn2_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[1]*ladder_cost(z1_clean, z_hat_bn1_noisy, y_weights_decoder_out_noisy)]
costs += [lambdas[0]*ladder_cost(z0_clean, z_hat_bn0_noisy, y_weights_decoder_out_noisy)]

cost = sum(costs)
# prediction passes
collect_out = lasagne.layers.get_output(
    l_y, sym_x, deterministic=True, collect=True)


# Get list of all trainable parameters in the network.
all_params = lasagne.layers.get_all_params(z_hat_bn0, trainable=True)
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
                          [cost, y_out_enc_noisy] + costs,
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
        x_batch = np.concatenate([x_train_lab, x_unsup], axis=0)

        # nb same targets all the time...
        output = f_train(x_batch, targets_train_lab)
        batch_loss, net_out = output[0], output[1]
        layer_costs = output[2:]
        # cut out preds with labels
        net_out = net_out[:num_labels]

        preds = np.argmax(net_out, axis=-1)
        confusion_train.batchadd(targets_train_lab, preds)
        losses += [batch_loss]
    return confusion_train, losses, layer_costs

# select correct training function...

def test_epoch(x, y):
    confusion_valid = parmesan.utils.ConfusionMatrix(num_classes)
    _ = f_collect(x_train)
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

