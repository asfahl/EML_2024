import torch
import numpy

# jax model: https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py
class MixerLayer( torch.nn.Module ):
    def __init__( self,
                  hidden_size_c,
                  seq_len_s,
                  mlp_dim_dc,
                  mlp_dim_ds ):
        super( MixerLayer, self ).__init__()

        self.ln1 = torch.nn.LayerNorm( hidden_size_c )
        self.ln2 = torch.nn.LayerNorm( hidden_size_c )

        self.mlp_tokens = torch.nn.Sequential( torch.nn.Linear( seq_len_s, mlp_dim_ds ),
                                               torch.nn.GELU(),
                                               torch.nn.Linear( mlp_dim_ds, seq_len_s ) )

        self.mlp_channels = torch.nn.Sequential( torch.nn.Linear( hidden_size_c, mlp_dim_dc ),
                                                 torch.nn.GELU(),
                                                 torch.nn.Linear( mlp_dim_dc, hidden_size_c ) )

    def forward( self, x ):
        x = x + self.mlp_tokens( self.ln1( x ).transpose( 1, 2 ) ).transpose( 1, 2 )
        x = x + self.mlp_channels( self.ln2( x ) )
        return x
    
class Mixer( torch.nn.Module ):
    def __init__( self,
                  hidden_size_c,
                  seq_len_s,
                  mlp_dim_dc,
                  mlp_dim_ds,
                  num_layers ):
        super( Mixer, self ).__init__()

        self.stem = torch.nn.Conv2d( 3,
                                     hidden_size_c,
                                     16,
                                     16 )

        self.mixer_layers = torch.nn.ModuleList( [ MixerLayer( hidden_size_c,
                                                               seq_len_s,
                                                               mlp_dim_dc,
                                                               mlp_dim_ds ) for _ in range( num_layers ) ] )

        self.ln = torch.nn.LayerNorm( hidden_size_c )
        self.head = torch.nn.Linear( hidden_size_c, 1000 )

    def load_parameters( self,
                         path_parameters ):
        pars = numpy.load( path_parameters )

        self.stem.load_state_dict( { 'weight': torch.tensor( pars['stem/kernel'] ).permute( 3, 2, 0, 1 ),
                                     'bias':   torch.tensor( pars['stem/bias'] ) } )
        
        for la in range( len( self.mixer_layers ) ):
            self.mixer_layers[la].load_state_dict( { 'ln1.weight':            torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_0/scale'] ),
                                                     'ln1.bias':              torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_0/bias'] ),
                                                     'ln2.weight':            torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_1/scale'] ),
                                                     'ln2.bias':              torch.tensor( pars[f'MixerBlock_{la}/LayerNorm_1/bias'] ),
                                                     'mlp_tokens.0.weight':   torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_0/kernel'] ).transpose( 0, 1 ),
                                                     'mlp_tokens.0.bias':     torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_0/bias'] ),
                                                     'mlp_tokens.2.weight':   torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_1/kernel'] ).transpose( 0, 1 ),
                                                     'mlp_tokens.2.bias':     torch.tensor( pars[f'MixerBlock_{la}/token_mixing/Dense_1/bias'] ),
                                                     'mlp_channels.0.weight': torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_0/kernel'] ).transpose( 0, 1 ),
                                                     'mlp_channels.0.bias':   torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_0/bias'] ),
                                                     'mlp_channels.2.weight': torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_1/kernel'] ).transpose( 0, 1 ),
                                                     'mlp_channels.2.bias':   torch.tensor( pars[f'MixerBlock_{la}/channel_mixing/Dense_1/bias'] ),
                                                   } )
            
        self.ln.load_state_dict( { 'weight': torch.tensor( pars['pre_head_layer_norm/scale'] ),
                                   'bias':   torch.tensor( pars['pre_head_layer_norm/bias'] ) } )
        self.head.load_state_dict( { 'weight': torch.tensor( pars['head/kernel'] ).transpose( 0, 1 ),
                                     'bias':   torch.tensor( pars['head/bias'] ) } )

    def forward( self, x ):
        x = self.stem( x )
        x = x.view( x.size(0), x.size(1), -1 ).transpose( 1, 2 )

        for mixer_layer in self.mixer_layers:
            x = mixer_layer( x )

        x = self.ln( x )
        x = x.mean( 1 )
        x = self.head( x )

        return x