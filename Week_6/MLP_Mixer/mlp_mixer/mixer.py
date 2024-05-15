import torch
import torchvision
import time
import pprint

import models.torchnn
import models.torchfunc

if __name__ == '__main__':
    #
    # config
    #
    # B/16
    config = { 'path_parameters': '/mnt/hd1/data/mixer_models/imagenet1k/imagenet1k_Mixer-B_16.npz',
               'hidden_size_c':   768,
               'seq_len_s':       196,
               'mlp_dim_dc':      3072,
               'mlp_dim_ds':      384,
               'num_layers':      12,
               'path_data':       '/mnt/hd1/data/imagenet/ilsvrc_2012',
               'batch_size':      64,
               'num_synthetic':   1,
               'model_impl':      'torchfunc',
               'dtype':           torch.float32,
               'cuda':            False }

    # L/16
    # config = { 'path_parameters': '/mnt/hd1/data/mixer_models/imagenet1k/imagenet1k_Mixer-L_16.npz',
    #            'hidden_size_c':   1024,
    #            'seq_len_s':       196,
    #            'mlp_dim_dc':      4096,
    #            'mlp_dim_ds':      512,
    #            'num_layers':      24,
    #            'path_data':       '/mnt/hd1/data/imagenet/ilsvrc_2012',
    #            'batch_size':      32,
    #            'num_synthetic':   1,
    #            'model_impl':      'torchnn',
    #            'dtype':           torch.float32,
    #            'cuda':            False }

    print( "*************************" )
    print( "*** Running MLP Mixer ***" )
    print( "*************************" )
    pprint.pprint( config )

    #
    # model
    #
    print( 'Creating model' )
    if config['model_impl'] == 'torchnn':
        mixer = models.torchnn.Mixer( hidden_size_c = config['hidden_size_c'],
                                      seq_len_s     = config['seq_len_s'],
                                      mlp_dim_dc    = config['mlp_dim_dc'],
                                      mlp_dim_ds    = config['mlp_dim_ds'],
                                      num_layers    = config['num_layers'] )
    elif config['model_impl'] == 'torchfunc':
        mixer = models.torchfunc.Mixer( num_layers = config['num_layers'] )
    else:
        raise ValueError( "Unknown model implementation" )

    print( 'Loading parameters' )
    mixer.load_parameters( config['path_parameters'] )

    #
    # dataloader
    #
    print( 'Setting up data loader' )
    trafo = torchvision.transforms.Compose([
                torchvision.transforms.Resize( 256 ),
                torchvision.transforms.CenterCrop( 224 ),
                torchvision.transforms.ToTensor(),
                # scale to [-1, 1]
                torchvision.transforms.Lambda( lambda x: 2.0 * (x - 0.5) ) ])

    #dataset_val = torchvision.datasets.ImageFolder( config['path_data'],
    #                                                trafo )
    dataset_val = torchvision.datasets.ImageNet( config['path_data'],
                                                 split = 'val',
                                                 transform = trafo )

    loader_val = torch.utils.data.DataLoader( dataset_val,
                                              batch_size = config['batch_size'],
                                              shuffle    = False )

    #
    # prep model
    #
    print( "Preparing model for execution" )
    # prep model
    mixer.eval()

    print( "***********************************************" )
    print( "*** Running synthetic performance benchmark ***" )
    print( "***********************************************" )
    # warm-up
    print( "Warming up.." )
    batch_synth = torch.randn( config['batch_size'], 3, 224, 224, dtype = config['dtype'] )

    for _ in range( max( config['num_synthetic'] // 10, 1 ) ):
        with torch.no_grad():
            mixer( batch_synth )

    # benchmark
    print( "Benchmarking.." )
    time_start = time.time()
    for _ in range( config['num_synthetic'] ):
        with torch.no_grad():
            mixer( batch_synth )
    time_end = time.time()
    duration_batch_synth = time_end - time_start
    duration_batch_synth /= config['num_synthetic']
    print( "  Time per batch: ", duration_batch_synth )
    print( "  Time per sample:", duration_batch_synth / config['batch_size'] )

    print( "*****************************************************" )
    print( "*** Running inference on ImageNet validation data ***" )
    print( "*****************************************************" )
    # inference
    num_samples = 0
    num_top1_correct = 0
    num_top5_correct = 0

    for id, data in enumerate( loader_val ):
        time_start = time.time()

        batch, labels = data

        with torch.no_grad():
            output = mixer( batch )

        num_samples += len(labels)
        num_top1_correct += (output.argmax(-1) == labels).sum().item()
        num_top5_correct += (output.topk(5, dim=1).indices == labels.unsqueeze(1)).sum().item()

        time_end = time.time()

        duration_batch = time_end - time_start

        if id % 10 == 0:
            print( "  Finished batch / sample:", id, "/", num_samples )
            print( "    Time per batch: ", duration_batch )
            print( "    Time per sample:", duration_batch / len(labels) )

            print( "    Top-1 accuracy: ", num_top1_correct / num_samples )
            print( "    Top-5 accuracy: ", num_top5_correct / num_samples )

    print( "  Samples:", num_samples )
    print( "  Top-1 accuracy:", num_top1_correct / num_samples )
    print( "  Top-5 accuracy:", num_top5_correct / num_samples )