from collections import OrderedDict


def revise_ckpt(state_dict, add_image_bbn_name=False, rn101=False, **kwargs):
    tmp_k = list(state_dict.keys())[0]
    print(f'Use revise_ckpt, tmp_k: {tmp_k}')
    if not tmp_k.startswith('module.') or tmp_k.startswith('img_backbone.'):
        to_add = 'module.'
        if add_image_bbn_name and not tmp_k.startswith('img_backbone.'):
            to_add += 'img_backbone.'
        state_dict = OrderedDict(
            {(to_add + k): v
             for k, v in state_dict.items()})
    elif add_image_bbn_name:
        state_dict = OrderedDict({k.replace('module.', 'module.img_backbone.'): v for k, v in state_dict.items()})
    return state_dict


def revise_ckpt_linear_probe(state_dict, ddp):
    tmp_k = list(state_dict.keys())[0]
    print(f'Use revise_ckpt_linear_probe, ddp: {ddp}, tmp_k: {tmp_k}')
    if ddp:
        if not tmp_k.startswith('module.'):
            state_dict = OrderedDict(
                {('module.' + k): v
                 for k, v in state_dict.items()})
    else:
        if tmp_k.startswith('module.'):
            state_dict = OrderedDict(
                {(k.replace('module.', '')): v
                 for k, v in state_dict.items()})
    return state_dict


def revise_ckpt_2(state_dict):
    param_names = list(state_dict.keys())
    for param_name in param_names:
        if 'img_neck.lateral_convs' in param_name or 'img_neck.fpn_convs' in param_name:
            del state_dict[param_name]
    return state_dict
