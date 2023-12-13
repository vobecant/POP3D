import argparse
import copy

import clip
import torch
import os

nuscenes_classes = ['adult pedestrian', 'child pedestrian', 'wheelchair', 'stroller', 'personal mobility',
                    'police officer', 'construction worker', 'animal', 'car', 'motorcycle', 'bicycle', 'bendy bus',
                    'rigid bus', 'truck', 'construction vehicle', 'ambulance vehicle', 'police car', 'trailer',
                    'barrier', 'traffic cone', 'debris', 'bicycle rack', 'driveable surface', 'sidewalk', 'terrain',
                    'other flat', 'manmade', 'vegetation']
nuscenes16_classes = ['barrier', 'bicycle', 'bus', 'car', 'constuction vehicle', 'motorcycle', 'pedestrian',
                      'traffic cone', 'trailer', 'truck', 'driveable surface', 'other flat', 'sidewalk', 'terrain',
                      'manmade', 'vegetation']

nuscenes_subcategories = {
    'noise': [
        'Any lidar return that does not correspond to a physical object, such as dust, vapor, noise, fog, raindrops, smoke and reflections.'],
    'animal': ["animal", 'cat', 'dog', 'rat', 'deer', 'bird'],
    'human.pedestrian.adult': ['Adult.'],
    'human.pedestrian.child': ['Child.'],
    'human.pedestrian.construction_worker': ['Construction worker'],
    'human.pedestrian.personal_mobility': ['skateboard', 'segway'],
    'human.pedestrian.police_officer': ['Police officer.'],
    'human.pedestrian.stroller': ['Stroller'],
    'human.pedestrian.wheelchair': ['Wheelchair'],
    'movable_object.barrier': ['Temporary road barrier to redirect traffic.', 'concrete barrier', 'metal barrier',
                               'water barrier'],
    'movable_object.debris': ['Movable object that is left on the driveable surface.', 'tree branch', 'full trash bag'],
    'movable_object.pushable_pullable': ['Object that a pedestrian may push or pull.', 'dolley', 'wheel barrow',
                                         'garbage-bin', 'shopping cart'],
    'movable_object.trafficcone': ['traffic cone.'],
    'static_object.bicycle_rack': ['Area or device intended to park or secure the bicycles in a row.'],
    'vehicle.bicycle': ['Bicycle'],
    'vehicle.bus.bendy': ['Bendy bus'],
    'vehicle.bus.rigid': ['Rigid bus'],
    'vehicle.car': ['Vehicle designed primarily for personal use.', 'car', 'vehicle', 'sedan', 'hatch-back', 'wagon',
                    'van', 'mini-van', 'SUV', 'jeep'],
    'vehicle.construction': ['Vehicle designed for construction.', 'crane'],
    'vehicle.emergency.ambulance': ['ambulance', 'ambulance vehicle'],
    'vehicle.emergency.police': ['police vehicle', 'police car', 'police bicycle', 'police motorcycle'],
    'vehicle.motorcycle': ['motorcycle', 'vespa', 'scooter'],
    'vehicle.trailer': ['trailer', 'truck trailer', 'car trailer', 'bike trailer'],
    'vehicle.truck': ['Vehicle primarily designed to haul cargo.', 'pick-up', 'lorry', 'truck', 'semi-tractor'],
    'flat.driveable_surface': ['Paved surface that a car can drive.', 'Unpaved surface that a car can drive.'],
    'flat.other': ['traffic island', 'delimiter', 'rail track', 'small stairs', 'lake', 'river'],
    'flat.sidewalk': ['sidewalk', 'pedestrian walkway', 'bike path'],
    'flat.terrain': ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    'static.manmade': ['man-made structure', 'building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant',
                       'flag', 'banner', 'street sign', 'electric circuit box', 'traffic light', 'parking meter',
                       'stairs'],
    'static.other': [
        'Points in the background that are not distinguishable, or objects that do not match any of the above labels.'],
    'static.vegetation': ['bushes', 'bush', 'plants', 'plant', 'potted plant', 'tree', 'trees'],
    'vehicle.ego': [
        'The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image.'],
}
cls2idx = [
    0, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 12, 13, 14, 15,
    16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22,
    23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 30, 30, 30, 30, 30, 30, 30, 31
]
nusc31to16 = CLASS_MAP = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16
}


nuscenes_subcategories_list = [v for v in nuscenes_subcategories.values()]
nuscenes_subcategories_list_flatten = [item for sublist in nuscenes_subcategories_list for item in sublist]


nuscenes_subcategories_extended = copy.deepcopy(nuscenes_subcategories)
nuscenes_subcategories_extended['human.pedestrian.adult'].append('person')
nuscenes_subcategories_extended['movable_object.trafficcone'].append('cone')
nuscenes_subcategories_extended['static_object.bicycle_rack'].append('bicycle rack')
nuscenes_subcategories_extended['flat.driveable_surface'].append('road')
nuscenes_subcategories_extended_list = [v for v in nuscenes_subcategories_extended.values()]


prompt_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    'this is the {} in the scene.', 'this is one {} in the scene.',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engeering script')
    parser.add_argument('--model', default='ViT16', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'],
                        help='clip model name')
    parser.add_argument('--class-set', default=['nuscenes_subcategories'], nargs='+',
                        choices=['nuscenes', 'nuscenes16', 'nuscenes_subcategories', 'nuscenes_subcategories_extended'],
                        help='the set of class names')
    parser.add_argument('--no-prompt-eng', action='store_true', help='disable prompt engineering')
    parser.add_argument('--multiple-names', action='store_true',
                        help='there are multiple names describing a single category')
    parser.add_argument('--output_dir', type=str, default='../data')

    args = parser.parse_args()
    return args


def zeroshot_classifier(model_name, classnames, templates):
    model, preprocess = clip.load(model_name)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_nuscenes_subcategories(model_name, classnames_list, templates):
    model, preprocess = clip.load(model_name)
    with torch.no_grad():
        zeroshot_weights_dataset = []
        class_indices = []
        for idx, classnames in enumerate(classnames_list):
            zeroshot_weights_class = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights_class.append(class_embedding)
                class_indices.append(idx)
            zeroshot_weights_class = torch.stack(zeroshot_weights_class, dim=1).cuda()
            zeroshot_weights_dataset.append(zeroshot_weights_class)
        class_indices = torch.tensor(class_indices)
    zeroshot_weights_dataset = torch.cat(zeroshot_weights_dataset, dim=1)
    return zeroshot_weights_dataset, class_indices


if __name__ == '__main__':
    args = parse_args()

    classes = []
    all_set_name = ''
    name_mapping = {'nuscenes': nuscenes_classes, 'nuscenes16': nuscenes16_classes,
                    'nuscenes_subcategories': nuscenes_subcategories_list_flatten,
                    'nuscenes_subcategories_extended': nuscenes_subcategories_extended_list
                    }
    for set_name in args.class_set:
        if set_name in name_mapping:
            classes += name_mapping[set_name]
            all_set_name += '_{}'.format(set_name)
        if set_name in ['blur'] or args.no_prompt_eng:
            prompt_templates = ['a photo of a {}.']
    if type(classes[0]) == list:
        pass
    
    elif not args.multiple_names:
        # remove redundant classes
        classes = list(dict.fromkeys(classes))
    # remove the first underline
    all_set_name = all_set_name[1:]
    print(classes)

    print(f"{len(classes)} class(es), {len(prompt_templates)} template(s)")

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32',
                    'ViT16': 'ViT-B/16'}
    if args.multiple_names:
        zeroshot_weights = zeroshot_classifier_nuscenes_subcategories(name_mapping[args.model], classes,
                                                                      prompt_templates)
    else:
        zeroshot_weights = zeroshot_classifier(name_mapping[args.model], classes, prompt_templates)
        zeroshot_weights = zeroshot_weights.permute(1, 0).float()
        print(zeroshot_weights.shape)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert len(zeroshot_weights)==len(cls2idx)
    print(f'zeroshot_weights.shape: {zeroshot_weights.shape}')
    # mapping = {c:nusc31to16[c] for c in cls2idx}
    zeroshot_weights = (zeroshot_weights, cls2idx)

    output_path = os.path.join(output_dir, f'{all_set_name}_{args.model}_clip_text.pth')
    torch.save(zeroshot_weights, output_path)
    print(f'Saved to {output_path}')
