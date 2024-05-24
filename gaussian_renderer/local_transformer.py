import numpy as np
import itertools
from tqdm import tqdm
import torch
from tqdm import tqdm
import torch.nn as nn

# Arguments to change

'''
Make all local transformer operations
Make a local transformer
Then make a lightweight 2 layer MLP which takes in the features and position (pos encoded) and makes a delta transformation to the features i.e final_features = transformer_features + MLP features (to perturb the features approproately)
'''

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class LocalTransformerNetwork(nn.Module):
    
    def __init__(self, transformer_args):
        super().__init__()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model= transformer_args['d_model'],
            nhead= transformer_args['nhead'],
            dim_feedforward= transformer_args['dim_feedforward'],
            dropout= transformer_args['dropout'],
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=transformer_args['num_layers']
        )
        
    def forward(self, x, positional_encoding):
        x = x + positional_encoding
        x = self.transformer_encoder(x)
        return x
        
    
class LocalMLPDelta(nn.Module):
    
    def __init__(self, transformer_args): # This module always uses the hard coded fourier frequencies
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(transformer_args["d_model"] + transformer_args["multi_res_in_delta_network"]*2*3+3, transformer_args["d_model"]),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_args["d_model"], transformer_args["d_model"])
        ) 
        
        self.embedder, _ = get_embedder(transformer_args["multi_res_in_delta_network"],3)
        
    def forward(self, x, positions):
        pos_encoding = self.embedder(positions)
        
        # print("positional encoding", pos_encoding.shape)
        
        x = torch.cat([pos_encoding, x],dim=-1)

        # print("Position cat data", x.shape)
        
        x = self.mlp(x) # computes a feature perturbation 
        return x        
    
            
class PositionalEncodingMLP(nn.Module): # This is MLP for Transformer positional encoding
    
    # This module expects xyz as input and outputs something as the Transformer dim if learnable or multires freq if not learnable
    
    def __init__(self, transformer_args): 
        super().__init__()
        
        self.embedder, _ = get_embedder(transformer_args["multi_res_dimension_in_transformer_pos_encode"],3)
    
        self.module = nn.Sequential (
            nn.Linear(transformer_args["multi_res_dimension_in_transformer_pos_encode"]*3*2+3,transformer_args["d_model"]),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_args["d_model"], transformer_args["d_model"])
        )
        
        self.transformer_args=transformer_args
    
    def forward(self,x):
        
        x_out = self.embedder(x)
    
        if self.transformer_args["use_learnable"]:
            x_out = self.module(x_out)    # Makes this to d_model dimension     
        
        return x_out
        
    # Simple MLP which takes in unique voxel position and outputs a vector of appropriate dim as positional encoding

class LocalTransformerOperations(nn.Module):
    
    '''
    1. Take all Gaussian centres and Features
    2. Voxelize all gaussian centres to a grid position
    3. Make a new tensor of all unique grid positions
    4. For all gaussians belonging to the unique grid position, get the average feature into the new grid
    5. Once the new grid is established -> Transformer (later)
    6. Make a new grid of the initial gaussian size
    7. Put the correct feature back to the original positions
    '''
    
    def __init__(self, transformer_args) -> None: # pos encode MLP will be common across local and global operators
        super().__init__()
        
        self.voxel_size = transformer_args["voxel_size"] # We can get this variable from 3dgs states like distance bw each gaussian -> 10th %tile or lesser? ; For coarse grid we can go 50%tile for more?
        self.transformer_model = LocalTransformerNetwork(transformer_args)
        self.delta_mlp = LocalMLPDelta(transformer_args)
        self.pos_encode_mlp = PositionalEncodingMLP(transformer_args)

    def voxelize_sample(self, data=None): # Pass in gaussian centres and they get voxelized the to correct grid
        voxel_id = torch.round(data/self.voxel_size) # Gives a unique ID like x_grid, y_grid, z_grid numbers
        voxel_positions = voxel_id*self.voxel_size

        unique_rows, inverse_indices = torch.unique(voxel_positions, return_inverse=True, dim=0) # inverse indices mean, the original N dim tensor can be made by stacking the inverse_ind index of the unique tensor
        
        return voxel_positions, voxel_id, unique_rows, inverse_indices

    def find_matching_rows(self, data, target_tensor):
        
        mask = (data == target_tensor)
        row_mask = mask.all(dim=1)
        matching_indices = torch.where(row_mask)[0]
        
        return matching_indices


    def create_feature_as_transformer_input(self, gaussian_centres, gaussian_features):
        voxelized_positions, _, unique_voxels, inverse_indices = self.voxelize_sample(gaussian_centres) # Unique voxel gives the position in xyz space that holds the voxels
        aggregated_features = torch.zeros(unique_voxels.shape[0], gaussian_features.shape[-1]).to(gaussian_features.device) # N, f

        # print("Aggregating features ...")

        # for idx, unique_voxel_position in tqdm(enumerate(unique_voxels), total=len(unique_voxels)):
        for idx, unique_voxel_position in enumerate(unique_voxels):
        
            matching_indices = self.find_matching_rows(voxelized_positions,unique_voxel_position)
            features_to_aggregate = gaussian_features[matching_indices]
            
            summed_features = torch.sum(features_to_aggregate, dim=0)
            aggregated_features[idx] = summed_features
        
        # Now we have a n, F tensor which aggregated all unique features
        
        return aggregated_features, inverse_indices, unique_voxels # Track unique voxels to keep track of order in which agg features was populated
    
    def recast_features_back(self, gaussian_centres, transformed_features, inverse_indices):
        
        recasted_features = torch.zeros(gaussian_centres.shape[0], transformed_features.shape[-1]).to(transformed_features.device)
        
        # print("Recasting features ...")
        # inverse indices holds which idx of unique voxel has to populate which id of the recasted feature
        # for idx, idx_to_recast in tqdm(enumerate(inverse_indices), total=len(inverse_indices)):
        for idx, idx_to_recast in enumerate(inverse_indices):
            recasted_features[idx] = transformed_features[idx_to_recast]
            
        return recasted_features          
    
    def forward(self, gaussian_centres, gaussian_features):
        
        
        # print(f"Gaussian features shape: {gaussian_features.shape}")
        
        aggregated_stuff, inv_indices, unique_voxel_positions = self.create_feature_as_transformer_input(gaussian_centres, gaussian_features)
        
        # print(f"Aggregated feature shape: {aggregated_stuff.shape}; Gaussian centres shape: {gaussian_centres.shape}")
        
        # This aggregated_stuff needs positional encoding
        positional_encoding = self.pos_encode_mlp(unique_voxel_positions) # this will give a learnable positional encoding using the MLP
        
        # print(f"Positional encoding shape: {positional_encoding.shape}")
        
        transformer_op = self.transformer_model(aggregated_stuff.unsqueeze(0), positional_encoding.unsqueeze(0)) # 1,N,f
        
        recasted_features = self.recast_features_back(gaussian_centres, transformer_op.squeeze(0), inv_indices)
        
        # print(f"Recasted feature shape: {recasted_features.shape}")
        
        mlp_delta_output = self.delta_mlp(recasted_features, gaussian_centres) # Learns the perturbation for each gaussian feature; in: N, f -> N,f
        
        final_features = recasted_features + mlp_delta_output
        
        # print(f"Final feature shape: {final_features.shape}")
        
        return final_features # N, 256
        
        
'''
1. Only 256 transformer -> 1x1 conv upscale
2. Only MLP for 256 -> 1x1 conv upscale

# MlP vs T 

1. What is no transformer? -> Only MLP pre-splatting
2. If fixed positional encoding sin cosin vs Learnable MLP based encoding -> Here also check if sin,cosine embed 10 or 4; 4 should make more sense to hold a smooth underlying distribution
3. Without MLP Delta i.e only transformer vectors; fixed pos encoding in MLP delta and check 6,8,10; higher = less conitnuous
'''
        
if __name__ == "__main__":
    
    # lt = LocalTransformerOperations(voxel_size=0.09)
    
    # data = torch.rand(100,3)
    
    # op, ids, unique_rows, inverse_indices = lt.voxelize_sample(data)
    # print(op.shape)
    # print(op[:5])
    # print(ids[:5])
    # print(unique_rows[:5])
    # print(inverse_indices[:5])
    
    
    
    # data = torch.randint(0,2,(10,3))
    # print(data)
    
    # mask = (data == data[2])
    # row_mask = mask.all(dim=1)
    # matching_indices = torch.where(row_mask)[0]
    # matching_rows = data[matching_indices]
    
    # print(matching_indices)
    # print(matching_rows)
    
    # data = torch.rand(5,100)
    # print(torch.sum(data,dim=0).shape)
    
    # data = torch.rand(90_000, 3).cuda() * 2
    # features = torch.rand(90_000, 512).cuda()
    
    # data = torch.tensor([
    #     [1,0,0],
    #     [0,1,0],
    #     [0,0,1],
    #     [1,0,0],
    #     [0,0,1],
    #     [0,1,0]
    # ])
    
    # features = torch.tensor([
    #     [0,0,0],
    #     [1,1,1],
    #     [2,2,2],data
    #     [0,0,0],
    #     [1,1,1],
    #     [4,4,4],
    # ])
    
    # aggregated_stuff, inv_indices = lt.create_feature_as_transformer_input(data, features)

    # print(aggregated_stuff)
    # print(inv_indices)
    
    # recasted_features = lt.recast_features_back(data, aggregated_stuff, inv_indices)
    
    # print(recasted_features)
    
    # vanilla_embedder, out_size = get_embedder(10, 3)
    # data = torch.rand(1,3)
    
    # print(data)
    
    # embedded_data = vanilla_embedder(data)
    
    # print(embedded_data.shape)
    # print(embedded_data)
    
    transformer_args = {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "num_layers": 4,
        "use_learnable": True,
        "multi_res_dimension_in_transformer_pos_encode": 4,
        "multi_res_in_delta_network": 6,
        "voxel_size": 0.1
    }
    
    transformer = LocalTransformerNetwork(transformer_args).cuda()
    ppe_mlp = PositionalEncodingMLP(transformer_args).cuda()
    
    positions = torch.rand(10_000,3).cuda()
    
    positional_vectors = ppe_mlp(positions).unsqueeze(0)
    # print(positional_vectors.shape)
    
    data = torch.rand(10_000, 256).cuda().unsqueeze(0)
    
    transformer_output = transformer(data, positional_vectors)
    print(transformer_output.shape)
    
    # mlp_perturb = LocalMLPDelta(transformer_args).cuda()
    # op = mlp_perturb(transformer_output, positions)
    
    # print(op.shape)
    
    # local_operation = LocalTransformerOperations().cuda()
    # gaussian_centres = torch.rand(50_000, 3).cuda()
    # gaussian_features = torch.rand(50_000, 256).cuda()
    
    # op = local_operation(gaussian_centres, gaussian_features)
    
    # print(op.shape)
    pass