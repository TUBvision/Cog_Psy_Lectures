import Image
import matplotlib.pyplot as plt
import numpy as np

def image_to_array(fname, in_format = 'png'):
    """
    read specified image file (default: png), converts it to grayscale and into numpy array
    input:
    ------
    fname       - name of image file
    in_format   - extension (png default)
    output:
    -------
    numpy array
    """
    im = Image.open('%s.%s' %(fname, in_format)).convert('L')
    im_matrix = [ im.getpixel(( y, x)) for x in range(im.size[1]) for y in range(im.size[0])]
    im_matrix = np.array(im_matrix).reshape(im.size[1], im.size[0])
    
    return im_matrix


def array_to_image(stimulus_array = None, outfile_name = None, out_format = 'bmp'):
    """
    convert numpy array into image (default = '.bmp') in order to display it with vsg.vsgDrawImage
    input:
    ------
    stimulus_array  -   numpy array
    outfile_name    -   ''
    out_format      -   'bmp' (default) or 'png'
    output:
    -------
    image           -   outfile_name.out_format
    """
    im_row, im_col = stimulus_array.shape
    im_new = Image.new("L",(im_col, im_row))
    im_new.putdata(stimulus_array.flatten())
    im_new.save('%s.%s' %(outfile_name, out_format), format = out_format)


def normalize_image(stim_in, new_min = 1, new_max = 256):
    """
    scale image range from [old_min, old_max] to [new_min=1, new_max = 256]
    """
    stim = stim_in.copy()
    stim = stim - stim.min()
    stim = stim/float(stim.max())
    stim = stim * (new_max - new_min)
    stim = stim + new_min
    return stim.round()

if __name__ == "__main__":
    
    stimulus_type = ['aligned', 'misaligned', 'filled', 'noinducer']
    stimulus_shape = ['thin', 'fat']
    for s_shape in stimulus_shape:
        for s_type in stimulus_type:
            stim_in = image_to_array('%s_%s_10' %(s_shape, s_type))
            if s_type == 'noinducer':
                stim_norm = normalize_image(stim_in, 127, stim_in.max())
            else:
                stim_norm = normalize_image(stim_in, 1, stim_in.max())
            array_to_image(stim_norm, 'norm_%s_%s_10' %(s_shape, s_type))

