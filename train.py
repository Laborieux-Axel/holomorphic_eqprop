import jax.numpy as jnp 
import haiku as hk 
import json 
import time 
import sys 

from jax import local_device_count, random
from jax.tree_util import tree_map

from utils.data import tfds_mnist_mlp, get_tfds_loaders, save_model, load_model
from utils.data import printLog, print_summary, make_result_dir, split
from utils.data import save_data, one_hot, save_checkpoint, load_checkpoint

from utils.functions import init_lrs_wds_mmt, get_activation, cos_ann
from utils.functions import L2_dict



# the first arg is the path to the json containing the hyperparameters
if len(sys.argv)>=2:
    with open(sys.argv[1]) as f:
        args = json.load(f)

path = make_result_dir(args, sys.argv[0]) if args['save'] else None


# Saving command line
command_line = 'python ' + ' '.join(sys.argv)
printLog(command_line, path_to_result=path)
printLog('', path_to_result=path)



if args['model_type']=='xp_mlp':
    
    from models import mlp
    
    # Unpacking argument values    
    seed = args['seed']
    key = random.PRNGKey(seed)
    batch_size = args['batch_size']
    in_size = args['in_size']
    n_targets = args['n_targets']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    algo = args['algo']
    num_epochs = args['epochs']
    beta = args['beta']
    noise = args['noise']
    T1, T2, N = args['T1'], args['T2'], args['N']
    Ts = T1, T2, N
    activation = get_activation(args)

    train_loader, fulltrain, val_loader = tfds_mnist_mlp(batch_size, seed)

    model = mlp.xp_mlp(archi, activation, n_targets, seed, noise=noise)
    params = model.init_params(in_size)

    lrs = {}
    for l in params:
        lrs[l] = archi[l]['lr']

    
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)


    data = {'tr_acc': [], 'val_acc': [], 'tr_loss': [], 'val_loss': [],
            'epochs': [], 'mean_log_L2': [], 'epoch_time': [], 'path': path,
            'val_corr': [], 'tr_corr': [], 'beta': beta, 'noise': noise}

    last_iter = len(train_loader) - 1 
    
    for epoch in range(num_epochs):
        
        start_time = time.perf_counter()
        for idx, batch in enumerate(train_loader):
            if idx==last_iter:
                prev_params = params
                held_x, held_y = batch[0], one_hot(batch[1], n_targets)

            params, h1, key = \
                model.train_step(params, batch, Ts, beta, lrs, algo, key) 

        epoch_time = time.perf_counter() - start_time

        if algo=='oscEP':
            last_beta = beta * jnp.exp(- (2j * jnp.pi)/N)
            L2s = model.batched_eval_conv(prev_params, held_x, held_y, 
                                          h1, last_beta, True)
        else:
            L2s = model.batched_eval_conv(prev_params, held_x, held_y,
                                          h1, 0.0, False)

        mean_log_L2 = jnp.mean(jnp.log10(L2s))
        printLog(f'Mean log L2 = {mean_log_L2.item()}', path_to_result=path)
        
        train_acc, train_corr, train_size, train_loss = \
            model.evaluate(params, fulltrain, T1)
        val_acc, val_corr, val_size, val_loss = \
            model.evaluate(params, val_loader, T1)

        data['tr_size'] = train_size
        data['val_size'] = val_size
        data['epochs'].append(epoch)
        data['epoch_time'].append(epoch_time)
        data['mean_log_L2'].append(mean_log_L2)
        data['tr_acc'].append(train_acc)
        data['val_acc'].append(val_acc)        
        data['tr_loss'].append(train_loss)
        data['val_loss'].append(val_loss)        
        data['val_corr'].append(val_corr)
        data['tr_corr'].append(train_corr)

        print_summary(data)
        if path is not None:
            save_data(data, path, 'data')



elif args['model_type']=='xent_p_cnn':

    from models import cnn
    
    # Unpacking argument values    
    seed = args['seed']
    task = args['task']
    batch_size = args['batch_size']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    algo = args['algo']
    num_epochs = args['epochs']
    beta = args['beta']
    T1, T2, N = args['T1'], args['T2'], args['N']
    Ts = T1, T2, N
    activation = get_activation(args)
    

    n_devices = local_device_count() if args['parallel'] else 1
    print(f"Number of devices: {n_devices}")
    

    train_loader, val_loader, n_targets, in_size, in_chan = \
        get_tfds_loaders(task, batch_size, seed)

    chance = 1/n_targets

    model = cnn.xent_p_cnn(archi, activation, n_targets, seed, n_devices)

    model.print_model(in_size, in_chan, path)
    params = model.init_params(in_size, in_chan)

    lrs, init_lrs, final_lrs, wds, mmt = init_lrs_wds_mmt(archi, params)


    if args.get('load_checkpoint') is not None:
        name = args['load_checkpoint']
        params, mmt, starting_epoch, data = \
            load_checkpoint(params, name, path, n_devices)
        lrs = cos_ann(lrs, init_lrs, final_lrs, starting_epoch-1, num_epochs)
        path = data['path']
    else:
        data = {'tr_acc': [], 'val_acc': [], 'tr_loss': [], 'val_loss': [],
                'epochs': [], 'mean_log_L2': [], 'epoch_time': [], 'path': path,
                'val_acc5': [], 'tr_corr': [], 'val_corr': [], 'val_corr5': []}
        starting_epoch = 0


   
    last_iter = len(train_loader) - 1 
    
    for epoch in range(starting_epoch, num_epochs):
        
        dropped = 0
        train_size = 0
        train_corr = 0
        train_loss = 0
        start_time = time.perf_counter()

        for idx, batch in enumerate(train_loader):
            if idx==last_iter:
                prev_params = params
                held_x, held_y = batch

            params, mmt, h1, corr, loss, drop = \
                model.train_step(params, batch, Ts, beta, lrs, wds, mmt, algo) 

            dropped += int(jnp.unique(drop).item())
            train_size += batch[0].shape[0]
            train_corr += jnp.unique(corr).item()
            train_loss += jnp.unique(loss).item()

        epoch_time = time.perf_counter() - start_time

        if algo=='oscEP':
            last_beta = beta * jnp.exp(- (2j * jnp.pi)/N)
            L2s = model.evaluate_convergence(prev_params, held_x, held_y, 
                                             h1, last_beta, True)
        else:
            L2s = model.evaluate_convergence(prev_params, held_x, held_y,
                                             h1, 0.0, False)

        mean_log_L2 = jnp.mean(jnp.log10(L2s))
        pnorm = L2_dict(params)/jnp.sqrt(n_devices)

        printLog(f'Mean log L2 = {mean_log_L2.item()}', path_to_result=path)
        printLog(f'Total dropped = {dropped}', path_to_result=path)
        printLog(f'param norm = {pnorm.item()}', path_to_result=path)
    
        lrs = cos_ann(lrs, init_lrs, final_lrs, epoch, num_epochs)
        
        train_acc = train_corr/train_size
        train_loss = train_loss/train_size
        val_acc, val_acc5, val_corr, val_corr5, val_size, val_loss = \
            model.evaluate(params, val_loader, T1)


        data['tr_size'] = train_size
        data['val_size'] = val_size
        data['epochs'].append(epoch)
        data['epoch_time'].append(epoch_time)
        data['mean_log_L2'].append(mean_log_L2)
        data['tr_acc'].append(train_acc)
        data['tr_corr'].append(train_corr)
        data['val_acc'].append(val_acc) 
        data['val_corr'].append(val_corr)
        data['val_acc5'].append(val_acc5)
        data['val_corr5'].append(val_corr5)
        data['tr_loss'].append(train_loss)
        data['val_loss'].append(val_loss)        
        
        print_summary(data)

        save_data(data, path, 'data')

        if args['save_checkpoint']:
            save_checkpoint(params, args, mmt, data)







elif args['model_type']=='ff_cnn':
    
    from models.ff_cnn import ff_cnn
    
    # Unpacking argument values    
    seed = args['seed']
    batch_size = args['batch_size']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    num_epochs = args['epochs']
    n_devices = local_device_count() if args['parallel'] else 1
    print(f"Number of devices: {n_devices}")
    task = args['task']
    activation = get_activation(args)


    train_loader, val_loader, n_targets, in_size, in_chan = \
        get_tfds_loaders(task, batch_size, seed)

    model = ff_cnn(archi, activation, n_targets, seed, n_devices)
    params = model.init_params(in_size, in_chan)

    lrs, init_lrs, final_lrs, wds, mmt = init_lrs_wds_mmt(archi, params)
    
    # if a model name has been specified in the command line
    if len(sys.argv)==3:
        name = sys.argv[2]
        params = load_model(params, name, path)
     
    data = {'tr_acc': [], 'val_acc': [], 'tr_loss': [], 'val_loss': [],
            'epochs': [], 'path': path, 'tr_acc5': [], 'val_acc5': [], 
            'epoch_time': [], 'val_corr': [], 'tr_corr': [], 'tr_corr5': [],
            'val_corr5': []}
    
    for epoch in range(num_epochs):

        train_size = 0
        train_corr = 0
        train_corr5 = 0
        train_loss = 0
        start_time = time.perf_counter()

        for x, y in train_loader:
            params, mmt, corr, corr5, loss = \
                model.train_step(params, x, y, lrs, wds, mmt) 

            train_size += x.shape[0]
            train_corr += jnp.unique(corr).item()
            train_corr5 += jnp.unique(corr5).item()
            train_loss += jnp.unique(loss).item()

        epoch_time = time.perf_counter() - start_time
    
        lrs = cos_ann(lrs, init_lrs, final_lrs, epoch, num_epochs)

        train_acc = train_corr/train_size
        train_acc5 = train_corr5/train_size
        train_loss = train_loss/train_size

        val_acc, val_acc5, val_corr, val_corr5, val_size, val_loss = \
            model.evaluate(params, val_loader)

        data['tr_size'] = train_size
        data['val_size'] = val_size
        data['epochs'].append(epoch)
        data['epoch_time'].append(epoch_time)
        data['tr_acc'].append(train_acc)
        data['tr_acc5'].append(train_acc5)
        data['val_acc'].append(val_acc)        
        data['val_acc5'].append(val_acc5)        
        data['tr_loss'].append(train_loss)
        data['val_loss'].append(val_loss)        
        data['val_corr'].append(val_corr)
        data['val_corr5'].append(val_corr5)
        data['tr_corr'].append(train_corr)
        data['tr_corr5'].append(train_corr5)
        
        save_data(data, path, 'data')
        print_summary(data)





if args['save_model']:
    save_model(params, args)





