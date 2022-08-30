import numpy as np
import jax.numpy as jnp
import haiku as hk
from jax import local_device_count, pmap
from jax.tree_util import tree_map

import tensorflow as tf
import tensorflow_datasets as tfds

import os
import json
import pickle
from datetime import datetime




# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU') 




def make_result_dir(args, simu_type):
    simu_type = simu_type.split('.')[0]
    date = datetime.now().strftime('%Y-%m-%d')
    begin_time = datetime.now().strftime('%H-%M-%S')
    path = 'results/'+simu_type+'/'+date+'_'+begin_time+'_'+args['name']
    if not(os.path.exists(path)):
        os.makedirs(path)
    with open(path+'/hyperparameters.json', 'w') as json_file:
        json.dump(args, json_file, indent=4)
    return path


def printLog(*args, path_to_result=None, **kwargs):
    print(*args, **kwargs)
    if path_to_result:
        with open(path_to_result+'/log.txt','a') as file:
            print(*args, **kwargs, file=file)




def print_summary(data):

    path = data['path']
    epoch_time = data['epoch_time'][-1]
    epoch = data['epochs'][-1]
    train_corr = data['tr_corr'][-1]
    val_corr = data['val_corr'][-1]
    train_acc = data['tr_acc'][-1]
    val_acc = data['val_acc'][-1]
    train_size = data['tr_size']
    val_size = data['val_size']
    train_loss = data['tr_loss'][-1]
    val_loss = data['val_loss'][-1]

    epoch_info = "\nEpoch {} in {:0.2f} sec".format(epoch+1, epoch_time)
    tr_inf = "Train accuracy :\t{}\t({}/{})".format(round(train_acc, 3),
                                                    train_corr, train_size)

    tr_loss_info = "\t\tTrain loss :\t{}".format(round(train_loss, 4))
    ts_inf = "Val accuracy   :\t{}\t({}/{})".format(round(val_acc, 3),
                                                    val_corr, val_size)

    val_loss_info = "\t\tVal loss :\t{}".format(round(val_loss, 4))

    printLog(epoch_info, path_to_result=path)
    printLog(tr_inf+tr_loss_info, path_to_result=path)
    printLog(ts_inf+val_loss_info, path_to_result=path)

    if data.get('val_acc5') is not None:
        val_acc5 = data['val_acc5'][-1]
        val_corr5 = data['val_corr5'][-1]
        ts_inf = "Top-5 val acc  :\t{}\t({}/{})".format(round(val_acc5, 3),
                                                        val_corr5, val_size)
        printLog(ts_inf, path_to_result=path)
    printLog('\n', path_to_result=path)



def tfds_mnist_mlp(batch_size, seed):

    tf.random.set_seed(seed)
    data_dir = './datasets/tfds'
    num_pixels = 784
    
    def data_transform(x, y):
        x = x/255
        x = tf.reshape(x, (len(x), num_pixels))
        return x, y

    # as_supervised=True gives us the (image, label) as a tuple
    train_ds = tfds.load(name='mnist', split='train', download=True,
                         as_supervised=True, data_dir=data_dir)
    train_ds = train_ds.shuffle(len(train_ds))
    train_ds = train_ds.batch(batch_size).map(data_transform).prefetch(1)
    train_ds = tfds.as_numpy(train_ds)

    fulltrain_ds = tfds.load(name='mnist', split='train', download=True,
                             as_supervised=True, data_dir=data_dir)
    fulltrain_ds = fulltrain_ds.batch(10000).map(data_transform).prefetch(1)
    fulltrain_ds = tfds.as_numpy(fulltrain_ds)

    test_ds = tfds.load(name='mnist', split='test', download=True,
                        as_supervised=True, data_dir=data_dir)
    test_ds = test_ds.batch(10000).map(data_transform).prefetch(1)
    test_ds = tfds.as_numpy(test_ds)

    return train_ds, fulltrain_ds, test_ds



def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:,None] == jnp.arange(k), dtype)

def par_one_hot(*args, **kwargs):
    return pmap(one_hot, in_axes=(0,None),
                static_broadcasted_argnums=1)(*args, **kwargs)



def get_tfds_loaders(task, batch_size, seed):

    tf.random.set_seed(seed)
    data_dir = './datasets/tfds'

    if task=='mnist':

        n_targets = 10
        in_size, in_chan = 28, 1
        
        def data_transform(x, y):
            x = x/255
            return x, y
    
        train_ds = tfds.load(name='mnist', split='train', download=True,
                             as_supervised=True, data_dir=data_dir)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .batch(batch_size)
            .map(data_transform)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)
    
        test_ds = tfds.load(name='mnist', split='test', download=True,
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .batch(10000)
            .map(data_transform)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)
    
    elif task=='cifar10':
    
        n_targets = 10
        in_size, in_chan = 32, 3

        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.4914, 0.4822, 0.4465])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.2023, 1.5*0.1994, 1.5*0.2010])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label


        train_ds = tfds.load(name='cifar10', split='train', download=True,
                             as_supervised=True, data_dir=data_dir)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)
    
        test_ds = tfds.load(name='cifar10', split='test', download=True,
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(10000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)

    elif task=='cifar100':
        
        n_targets = 100
        in_size, in_chan = 32, 3

        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.4914, 0.4822, 0.4465])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.2023, 1.5*0.1994, 1.5*0.2010])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label

        # cifar100_data = tfds.load(name="cifar100", batch_size=-1,
                                  # data_dir=data_dir, download=True) 

        train_ds = tfds.load(name='cifar100', split='train', download=True,
                             as_supervised=True, data_dir=data_dir)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)

        test_ds = tfds.load(name='cifar100', split='test', download=True,
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(10000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)


    elif task=='imagenet32':
    
        n_targets = 1000
        in_size, in_chan = 32, 3

        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.485, 0.456, 0.406])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.229, 1.5*0.224, 1.5*0.225])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label

        split = tfds.split_for_jax_process('train', drop_remainder=True)

        train_ds = tfds.load(name='imagenet_resized/32x32', split=split,
                             as_supervised=True, data_dir=data_dir, 
                             shuffle_files=True, download=True)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size, drop_remainder=True)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)
    
        test_ds = tfds.load(name='imagenet_resized/32x32', split='validation',
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(10000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)


    return train_ds, test_ds, n_targets, in_size, in_chan 






def save_model(params, args):
    params = hk.data_structures.to_mutable_dict(params)
    name = args['name']
    if args.get('parallel') is not None and args['parallel']:
        params = unsplit_params(params)
    if not(os.path.exists('./saved_models')):
        os.makedirs('./saved_models')
    with open('./saved_models/'+name+'.pickle', 'wb') as f:
        pickle.dump(params, f)

def load_model(params, name, path):
    model_path = './saved_models/'+name+'.pickle'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
            params = hk.data_structures.to_immutable_dict(params)
        printLog('Model successfully loaded', path_to_result=path)
    else:
        printLog('No model found in saved_models dir.', path_to_result=path)
    return params







def save_checkpoint(params, args, mmt, data):
    checkpoint = {}
    params = hk.data_structures.to_mutable_dict(params)
    mmt = hk.data_structures.to_mutable_dict(mmt)
    name = args['name']
    if args.get('parallel') is not None and args['parallel']:
        params = unsplit_params(params)
        mmt = unsplit_params(mmt)
    checkpoint['params'] = params
    checkpoint['mmt'] = mmt
    checkpoint['data'] = data

    if not(os.path.exists('./checkpoint')):
        os.makedirs('./checkpoint')
    with open('./checkpoint/'+name+'.pickle', 'wb') as f:
        pickle.dump(checkpoint, f)



def load_checkpoint(params, name, path, n_devices):
    checkpoint_path = './checkpoint/'+name+'.pickle'
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            params = checkpoint['params']
            params = hk.data_structures.to_immutable_dict(params)
            mmt = checkpoint['mmt']
            data = checkpoint['data']
            starting_epoch = data['epochs'][-1] + 1
            if n_devices>1:
                params = tree_map(lambda x: jnp.array([x]*n_devices), params)
                mmt = tree_map(lambda x: jnp.array([x]*n_devices), mmt)
        printLog('Checkpoint successfully loaded', path_to_result=path)
    else:
        printLog('No checkpoint found', path_to_result=path)
    return params, mmt, starting_epoch, data




def save_data(data, path, name):
    if not(os.path.exists(path)):
        os.makedirs(path)
    with open(path+'/'+name+'.pickle', 'wb') as f:
        pickle.dump(data, f)

def load_data(path, name):
    if not(os.path.exists(path)):
        print('no such path')
        return
    else:
        complete_path = path+'/'+name+'.pickle'
    with open(complete_path, 'rb') as f:
        data = pickle.load(f)
    return data





def split(arr):
    """Splits the first axis of `arr` evenly across the number of devices."""
    n_devices = local_device_count()
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

def unsplit(arr):
    return arr.reshape(arr.shape[0]*arr.shape[1], *arr.shape[2:])

def unsplit_params(p):
    return tree_map(lambda x: x[0,:], p)

def split_batch(batch):
    return split(batch[0]), split(batch[1])





