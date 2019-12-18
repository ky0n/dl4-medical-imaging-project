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
    return i.get_data(), i.affine[0, 0], i.affine[1, 1], i.affine[2, 2]

def load_segmentation_mask(filename):
    i = nibabel.load(filename)
    print(i.get_data().shape)
    print(repr(i))


def make_image_slices(image, nr_slices, colours=True):

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
    if colours:
        imagestrip_rgb = np.stack([
            np.maximum(np.minimum(imagestrip * 2 - .5, 1), 0),
            np.maximum(imagestrip * 2 - 1, 0),
            np.maximum(1 - imagestrip * 2, 0)
        ], axis=2)
    else:
        imagestrip_rgb = np.stack([imagestrip] * 3, axis=2)

    return imagestrip_rgb


def show_image_collage(image, nr_slices, sx, sy, sz, labels=None):

    # front, side and top view
    view1 = image
    view2 = np.moveaxis(view1, 2, 0)
    view3 = np.moveaxis(view2, 2, 0)
    views = [view1, view2, view3]
    aspects = [sy / sx, sz / sx, sy / sz]
    slice_views = [make_image_slices(v, nr_slices, labels is None) for v in views]

    if labels is not None:
        label_view1 = labels
        label_view2 = np.moveaxis(label_view1, 2, 0)
        label_view3 = np.moveaxis(label_view2, 2, 0)
        label_views = [label_view1, label_view2, label_view3]
        label_slice_views = [make_image_slices(v, nr_slices) for v in label_views]
        slice_views = [0.5*a + 0.5*b for a, b in zip(slice_views, label_slice_views)]

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.imshow(slice_views[i], aspect=aspects[i])
    plt.show()


def load_dataset(base_path):
    images_tr_path = os.path.join(base_path, "imagesTr")
    images_ts_path = os.path.join(base_path, "imagesTs")
    labels_tr_path = os.path.join(base_path, "labelsTr")
    images_tr = {f: os.path.join(images_tr_path, f) for f in os.listdir(images_tr_path) if f.endswith(".nii.gz") and not f.startswith(".")}
    labels_tr = {f: os.path.join(labels_tr_path, f) for f in os.listdir(labels_tr_path) if f.endswith(".nii.gz") and not f.startswith(".")}
    images_ts = {f: os.path.join(images_ts_path, f) for f in os.listdir(images_ts_path) if f.endswith(".nii.gz") and not f.startswith(".")}
    tr = {f: (images_tr[f], labels_tr[f]) for f in images_tr if f in labels_tr}
    return tr, images_ts

# Daten liegen im data Ordner, der sich im roo des Projektes befindet.
def main():
    print("Gude!")

    image, sx, sy, sz = load_image(LIVER_PATH + TRAINING + 'liver_0.nii.gz')
    segmentation, _, _, _ = load_image(LIVER_PATH + LABELS + 'liver_0.nii.gz')
    show_image_collage(image, 5, sx, sy, sz, segmentation)

    image, sx, sy, sz = load_image(LIVER_PATH + TEST + 'liver_132.nii.gz')
    show_image_collage(image, 5, sx, sy, sz)

    all_files = load_dataset(LIVER_PATH)
    print(repr(all_files))

main()