level = 5
corrupts = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'gaussian_blur', 'defocus_blur', 'spatter', 'jpeg_compression']
# template = 'maskclip_vit16_480x480_pascal_context_59_template.py'
template = 'maskclip_r50_480x480_pascal_context_59_template.py'

for corrupt in corrupts:
    with open(template, 'r') as f:
        save_name = template.replace('template', corrupt+str(level))
        with open(save_name, 'w') as out:
            for line in f.readlines():
                line = line.replace('ANCHOR1', corrupt)
                line = line.replace('ANCHOR2', str(level))
                out.write(line)