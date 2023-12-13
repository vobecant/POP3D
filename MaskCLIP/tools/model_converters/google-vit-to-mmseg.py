# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import OrderedDict

import torch
import numpy as np

def convert(src, dst, num_layers, prefix):
    # load numpy model
    assert os.path.isfile(src), f'no checkpoint found at {src}'
    with open(src, 'rb') as f:
        blobs = np.load(f, allow_pickle=False)
        keys, weights = zip(*list(blobs.items()))
        src_state_dict = {key: weight for key, weight in zip(keys, weights)}
    
    # convert to pytorch style
    dst_state_dict = OrderedDict()

    dst_state_dict[f'{prefix}cls_token'] = torch.from_numpy(
        src_state_dict['cls'])
    dst_state_dict[f'{prefix}pos_embed'] = torch.from_numpy(
        src_state_dict['Transformer/posembed_input/pos_embedding'])
    dst_state_dict[f'{prefix}patch_embed.projection.weight'] = torch.from_numpy(
        src_state_dict['embedding/kernel']).permute(3, 2, 0, 1)
    dst_state_dict[f'{prefix}patch_embed.projection.bias'] = torch.from_numpy(
        src_state_dict['embedding/bias'])
    dst_state_dict[f'{prefix}ln1.weight'] = torch.from_numpy(
        src_state_dict['Transformer/encoder_norm/scale'])
    dst_state_dict[f'{prefix}ln1.bias'] = torch.from_numpy(
        src_state_dict['Transformer/encoder_norm/bias'])
    if 'pre_logits/kernel' in src_state_dict:
        dst_state_dict['head.layers.pre_logits.weight'] = torch.from_numpy(
            src_state_dict['pre_logits/kernel']).t()
        dst_state_dict['head.layers.pre_logits.bias'] = torch.from_numpy(
            src_state_dict['pre_logits/bias'])

    for i in range(num_layers):
        dst_state_dict[f'{prefix}layers.{i}.ln1.weight'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])
        dst_state_dict[f'{prefix}layers.{i}.ln1.bias'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        dst_state_dict[f'{prefix}layers.{i}.ln2.weight'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])
        dst_state_dict[f'{prefix}layers.{i}.ln2.bias'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])

        dst_state_dict[f'{prefix}layers.{i}.ffn.layers.0.0.weight'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel']).t()
        dst_state_dict[f'{prefix}layers.{i}.ffn.layers.0.0.bias'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        dst_state_dict[f'{prefix}layers.{i}.ffn.layers.1.weight'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel']).t()
        dst_state_dict[f'{prefix}layers.{i}.ffn.layers.1.bias'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

        dst_state_dict[f'{prefix}layers.{i}.attn.attn.out_proj.weight'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'])\
                .permute(2, 0, 1).flatten(start_dim=1)
        dst_state_dict[f'{prefix}layers.{i}.attn.attn.out_proj.bias'] = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])

        q_weight = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'])\
                .flatten(start_dim=1).t()
        k_weight = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'])\
                .flatten(start_dim=1).t()
        v_weight = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'])\
                .flatten(start_dim=1).t()
        dst_state_dict[f'{prefix}layers.{i}.attn.attn.in_proj_weight'] = torch.cat([q_weight, k_weight, v_weight])
        
        q_bias = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'])\
                .flatten()
        k_bias = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'])\
                .flatten()
        v_bias = torch.from_numpy(
            src_state_dict[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'])\
                .flatten()
        dst_state_dict[f'{prefix}layers.{i}.attn.attn.in_proj_bias'] = torch.cat([q_bias, k_bias, v_bias])

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = dst_state_dict
    torch.save(checkpoint, dst)
    # for k, v in dst_state_dict.items():
    #     print(f'{k}: {v.shape}')
    # print()
    # for k, v in src_state_dict.items():
    #     print(f'{k}: {v.shape}')

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src numpy model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument(
        '--num-layers',
        type=int,
        choices=[12],
        default=12,
        help='number of ViT layers')
    parser.add_argument('--backbone', action='store_true', help='add backbone. to prefix')
    args = parser.parse_args()
    prefix = 'backbone.' if args.backbone else ''
    convert(args.src, args.dst, args.num_layers, prefix)

if __name__ == '__main__':
    main()