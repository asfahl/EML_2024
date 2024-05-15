import torch
import numpy

class MixerLayer():
    def __init__(self):
        self.tensors = {}

    def forward( self,
                 x ):
        tmp = torch.layer_norm( x,
                                normalized_shape = [ x.size()[-1] ],
                                weight = self.tensors['ln1.weight'],
                                bias   = self.tensors['ln1.bias'] )

        # mlp_tokens
        tmp = torch.matmul( self.tensors['mlp_tokens.0.weight'],
                            tmp )
        tmp = torch.add( tmp,
                         self.tensors['mlp_tokens.0.bias'] )
        tmp = torch.nn.functional.gelu( tmp )
        tmp = torch.matmul( self.tensors['mlp_tokens.2.weight'],
                            tmp )
        tmp = torch.add( tmp,
                         self.tensors['mlp_tokens.2.bias'] )
        x = torch.add( x, tmp )

        # mlp_channels
        tmp = torch.layer_norm( x,
                                normalized_shape = [ x.size()[-1] ],
                                weight = self.tensors['ln2.weight'],
                                bias   = self.tensors['ln2.bias'] )
        
        tmp = torch.matmul( tmp,
                            self.tensors['mlp_channels.0.weight'] )
        tmp = torch.add( tmp,
                         self.tensors['mlp_channels.0.bias'] )
        tmp = torch.nn.functional.gelu( tmp )
        tmp = torch.matmul( tmp,
                            self.tensors['mlp_channels.2.weight'] )
        tmp = torch.add( tmp,
                         self.tensors['mlp_channels.2.bias'] )
        x = torch.add( x, tmp )

        return x

class Mixer():
    def __init__( self,
                  num_layers ):
        self.tensors = {}
        self.mixer_layers = [ MixerLayer() for _ in range( num_layers ) ]
    
    def load_parameters( self,
                         path_parameters ):
        pars = numpy.load( path_parameters )

        self.tensors['stem.weight'] = torch.tensor( pars['stem/kernel'] )
        self.tensors['stem.weight'] = self.tensors['stem.weight'].view( -1, self.tensors['stem.weight'].size()[-1] )
        self.tensors['stem.bias']   = torch.tensor( pars['stem/bias'] )

        for la in range( len( self.mixer_layers ) ):
            self.mixer_layers[la].tensors = { 'ln1.weight':            torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_0/scale'] ),
                                              'ln1.bias':              torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_0/bias'] ),
                                              'ln2.weight':            torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_1/scale'] ),
                                              'ln2.bias':              torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_1/bias'] ),
                                              'mlp_tokens.0.weight':   torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_0/kernel'] ).transpose( 0, 1 ).unsqueeze(0).contiguous(),
                                              'mlp_tokens.0.bias':     torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_0/bias'] ).unsqueeze(-1),
                                              'mlp_tokens.2.weight':   torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_1/kernel'] ).transpose( 0, 1 ).unsqueeze(0).contiguous(),
                                              'mlp_tokens.2.bias':     torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_1/bias'] ).unsqueeze(-1),
                                              'mlp_channels.0.weight': torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_0/kernel'] ),
                                              'mlp_channels.0.bias':   torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_0/bias'] ),
                                              'mlp_channels.2.weight': torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_1/kernel'] ),
                                              'mlp_channels.2.bias':   torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_1/bias'] ) }

        self.tensors['ln.weight'] = torch.tensor( pars['pre_head_layer_norm/scale'] )
        self.tensors['ln.bias']   = torch.tensor( pars['pre_head_layer_norm/bias'] )
 
        self.tensors['head.weight'] = torch.tensor( pars['head/kernel'] )
        self.tensors['head.bias']   = torch.tensor( pars['head/bias'] )

    def eval( self ):
        pass

    def forward( self, x ):
        #           0          1          2              3   4              5
        x = x.view( x.size(0), x.size(1), x.size(2)//16, 16, x.size(3)//16, 16 )
        x = x.permute( 0, 2, 4, 3, 5, 1 ).contiguous()
        x = x.view( x.size(0), x.size(1)*x.size(2), -1 )

        x = torch.matmul( x, self.tensors['stem.weight'] )
        x = torch.add( x, self.tensors['stem.bias'] )

        for mixer_layer in self.mixer_layers:
            x = mixer_layer.forward( x )

        x = torch.layer_norm( x,
                              normalized_shape = [ x.size()[-1] ],
                              weight = self.tensors['ln.weight'],
                              bias   = self.tensors['ln.bias'] )
        x = torch.mean( x, dim = 1 )
        x = torch.matmul( x, self.tensors['head.weight'] )
        x = torch.add( x, self.tensors['head.bias'] )

        return x
    
    def __call__( self, x ):
        return self.forward( x )