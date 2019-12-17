from nilearn import plotting
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")       # base path of the git repo
LIVER_PATH = os.path.join(BASE_PATH, 'data/Task03_Liver/')
PROSTATE_PATH = os.path.join(BASE_PATH, 'data/Task05_Prostate/')


LIVER_PREFIX = 'liver_'
LIVER_POSTFIX = '.nii.gz'
PROSTATE_PREFIX = 'prostate_'
PROSTATE_POSTFIX = '.nii.gz'
TRAINING = 'imagesTr/'
TEST = 'imagesTs/'
LABELS = 'labelsTr/'


def show_image_from_file(path):
    print(path)
    plotting.plot_img(path)
    plotting.show()


def load_image(filename):
    print(filename)
    i = nibabel.load(filename)
    print(i.affine)
    return i.get_data(), i.affine[0, 0], i.affine[1, 1], i.affine[2, 2]


def make_image_volumetric(image: np.ndarray, h):
    # project the 3d image onto a 2d plane with rgb colors
    # using the emission-absorpion model.
    # and show the result

    # normalize values to be between 0 and 1
    max = image.max()
    min = image.min()
    image = (image - min) / (max - min)

    # create image, filled with some grey bg colour
    size_x, size_y, size_z = image.shape
    image_r = np.ones((size_x, size_y)) * .5
    image_g = np.ones((size_x, size_y)) * .5
    image_b = np.ones((size_x, size_y)) * .5

    # apply the layers from back to front
    for z in range(size_z):
        layer = image[:, :, z]
        dz = h / size_z
        absorption = np.exp(-(0.3 * layer) * dz)
        emission_r = .4 * layer * dz
        emission_g = 4 * layer ** 5 * dz
        emission_b = .2 * (1-layer) ** 5 * dz
        image_r = image_r * absorption + emission_r
        image_g = image_g * absorption + emission_g
        image_b = image_b * absorption + emission_b

    # show result
    return np.stack([image_r, image_g, image_b], axis=2)


def make_image_slices(image, nr_slices):

    # normalize values to be between 0 and 1
    img_max = image.max()
    img_min = image.min()
    image = (image - img_min) / (img_max - img_min)

    # calculate the distance between the slices
    size_x, size_y, size_z = image.shape
    slices_dist = size_z / (nr_slices + 1)

    # get the slices
    slices = [image[:, :, int(s * slices_dist)] for s in range(1, nr_slices + 1)]

    # concatenate and show
    imagestrip = np.concatenate(slices, axis=1)
    imagestrip_rgb = np.stack([
        np.maximum(np.minimum(imagestrip * 2 - .5, 1), 0),
        np.maximum(imagestrip * 2 - 1, 0),
        np.maximum(1 - imagestrip * 2, 0)
    ], axis=2)
    return imagestrip_rgb


def show_image_collage(image, nr_slices, sx, sy, sz):

    # front, side and top view
    view1 = image
    view2 = np.moveaxis(view1, 2, 0)
    view3 = np.moveaxis(view2, 2, 0)
    views = [view1, view2, view3]
    aspects = [sy / sx, sz / sx, sy / sz]
    sizes = [sz, sy, sx]

    volume_views = [make_image_volumetric(v, s) for v, s in zip(views, sizes)]
    slice_views = [make_image_slices(v, nr_slices) for v in views]
    complete_views = [np.concatenate([v, s], axis=1) for v, s in zip(volume_views, slice_views)]

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.imshow(complete_views[i], aspect=aspects[i])
    plt.show()


# Daten liegen im data Ordner, der sich im roo des Projektes befindet.
def main():
    print("Gude!")
    show_image_from_file(LIVER_PATH + TRAINING + 'liver_0.nii.gz')
    image, sx, sy, sz = load_image(LIVER_PATH + TRAINING + 'liver_0.nii.gz')
    show_image_collage(image, 5, sx, sy, sz)

main()