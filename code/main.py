import math
import pickle
from random import shuffle

from keras.utils import np_utils, plot_model
from nilearn import plotting
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import pad

from nnunet import make_nnunet
from unet import create_unet_3d, create_unet_3d_small, create_unet_3d_small_small

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")       # base path of the git repo
LIVER_PATH = os.path.join(BASE_PATH, 'data/Task03_Liver/')
PROSTATE_PATH = os.path.join(BASE_PATH, 'data/Task05_Prostate/')
TRAINING = 'imagesTr'
TEST = 'imagesTs'
LABELS = 'labelsTr'
CACHE_PATH = os.path.join(BASE_PATH, 'cache')
if not os.path.exists(CACHE_PATH):
    os.mkdir(CACHE_PATH)

# todo
# move cache path intoo repo
# - ask if model should be saved after abort
# - function to predict and then show the segmentation with trained model
# - implement intersection over union
# - The only notable changes to the original U-Net architecture are the use of padded convolutions to achieve identical output and input shapes, instance normalization and Leaky ReLUs instead of ReLUs

# results from the experiment_planning/plan_and_preprocess_task.py script in nnUNet.
options_liver = {
    0: {
        'batch_size': 2,
        'num_pool_per_axis': [5, 5, 5],
        'patch_size': [128, 128, 128],
        'median_patient_size_in_voxels': [195, 207, 207],
        'current_spacing': [2.473119, 1.89831205, 1.89831205],
        'original_spacing': [1., 0.76757812, 0.76757812],
        'do_dummy_2D_data_aug': False,
        'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    },
    1: {
        'batch_size': 2,
        'num_pool_per_axis': [5, 5, 5],
        'patch_size': [128, 128, 128],
        'median_patient_size_in_voxels': [482, 512, 512],
        'current_spacing': [1., 0.76757812, 0.76757812],
        'original_spacing': [1., 0.76757812, 0.76757812],
        'do_dummy_2D_data_aug': False,
        'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    }
}
options_prostate = {
    0: {
        'batch_size': 2,
        'num_pool_per_axis': [2, 6, 6],
        'patch_size': [20, 320, 256],
        'median_patient_size_in_voxels': [20, 320, 319],
        'current_spacing': [3.5999999, 0.625, 0.625],
        'original_spacing': [3.5999999, 0.625, 0.625],
        'do_dummy_2D_data_aug': True,
        'pool_op_kernel_sizes': [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
        'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    }
}


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


def one_hot_encode(segmentation):
    s1, s2, s3 = segmentation.shape
    encoded = np.zeros(shape=(s1, s2, s3, 3))
    encoded[:, :, :, 0][segmentation == 0] = 1
    encoded[:, :, :, 1][segmentation == 1] = 1
    encoded[:, :, :, 2][segmentation == 2] = 1
    return encoded


def rescale_image(image, sx, sy, sz, tx, ty, tz):
    cur_x, cur_y, cur_z, channels = image.shape
    new_x = int(cur_x * sx / tx)
    new_y = int(cur_y * sy / ty)
    new_z = int(cur_z * sz / tz)
    return resize(image, (new_x, new_y, new_z, channels))


def get_patches(image_size, patch_size, output_size):
    nr_patches = list(int(image_size[i] / output_size[i]) +
                      (1 if image_size[i] % output_size[i] > 0 else 0) for i in range(3))
    for x in range(nr_patches[0]):
        for y in range(nr_patches[1]):
            for z in range(nr_patches[2]):
                inp_min = [0] * 3
                inp_max = [0] * 3
                out_min = [0] * 3
                out_max = [0] * 3
                c = [x, y, z]
                for d in range(3):
                    out_min[d] = c[d] * output_size[d]
                    out_max[d] = out_min[d] + output_size[d]
                    if out_max[d] > image_size[d]:
                        out_max[d] = image_size[d]
                        out_min[d] = out_max[d] - output_size[d]

                    pad_sum = patch_size[d] - output_size[d]
                    pad_before = int(pad_sum / 2)
                    pad_after = pad_sum - pad_before
                    inp_min[d] = out_min[d] - pad_before
                    inp_max[d] = out_max[d] + pad_after
                yield tuple(inp_min), tuple(inp_max), tuple(out_min), tuple(out_max)


def make_train_examples(files, modality, patch_size, output_size, new_spacing):
    for name, (file_input, file_target) in files.items():
        cache_path = os.path.join(CACHE_PATH, name + "-cache")
        if os.path.exists(cache_path):
            print(cache_path)
            with open(cache_path, "rb") as f:
                image_input, image_target, image_size, input_padding_before, target_padding_before = pickle.load(f)
        else:
            # read image
            image_input, sx, sy, sz = load_image(file_input)

            # transpose
            image_input = np.transpose(image_input)
            tmp = sx
            sx = sz
            sz = tmp

            # select modality
            if image_input.ndim == 3:
                image_input = np.expand_dims(image_input, 3)
            image_input = image_input[:, :, :, modality:modality+1]

            # resize they all have a common spacing / scaling?
            #new_spacing = 0.76757812, 0.76757812, 1.    # median from the liver dataset
            #new_spacing = 2., 2., 2.    # actually, much smaller size so it fits into my ram
            image_input = rescale_image(image_input, sx, sy, sz, *new_spacing)

            # load image target, one-hot-encode it and apply the same scaling factor
            image_target, _, _, _ = load_image(file_target)
            image_target = np.transpose(image_target)
            image_target = one_hot_encode(image_target)
            image_target = rescale_image(image_target, sx, sy, sz, *new_spacing)

            # add padding to the image
            input_padding_before = [0] * 3
            input_padding_after = [0] * 3
            target_padding_before = [0] * 3
            image_size = image_input.shape[:3]
            for d in range(3):
                p1 = int((patch_size[d] - output_size[d]) / 2) + 1
                p2 = output_size[d] + p1 - image_size[d]
                input_padding_before[d] = max(p1, p2)
                input_padding_after[d] = p1
                p3 = output_size[d] - image_size[d]
                target_padding_before[d] = max(0, p3)
            input_padding = tuple(zip(input_padding_before + [0], input_padding_after + [0]))
            target_padding = tuple(zip(target_padding_before + [0], [0] * 4))
            image_input = pad(image_input, input_padding, mode='symmetric')
            image_target = pad(image_target, target_padding, mode='symmetric')

            with open(cache_path, 'wb') as f:
                pickle.dump((image_input, image_target, image_size, input_padding_before, target_padding_before, ), f)

        # generate image patches
        patch_positions = list(get_patches(image_size, patch_size, output_size))
        shuffle(patch_positions)
        for inp_min, inp_max, out_min, out_max in patch_positions:
            # get the actual pixel positions by adding the padding that was applied to the image
            inp_min = [inp_min[i] + input_padding_before[i] for i in range(3)]
            inp_max = [inp_max[i] + input_padding_before[i] for i in range(3)]
            out_min = [out_min[i] + target_padding_before[i] for i in range(3)]
            out_max = [out_max[i] + target_padding_before[i] for i in range(3)]

            # crop to the given rectangle
            input_patch = image_input[(inp_min[0]):(inp_max[0]), (inp_min[1]):(inp_max[1]), (inp_min[2]):(inp_max[2]), :]
            target_patch = image_target[(out_min[0]):(out_max[0]), (out_min[1]):(out_max[1]), (out_min[2]):(out_max[2]), :]
            yield input_patch, target_patch


def make_train_batches(files, modality, patch_size, output_size, batch_size, spacing):
    next_batch_inputs = []
    next_batch_targets = []
    for input, target in make_train_examples(files, modality, patch_size, output_size, spacing):
        next_batch_inputs.append(input)
        next_batch_targets.append(target)
        if len(next_batch_inputs) >= batch_size:
            yield np.stack(next_batch_inputs, axis=0), np.stack(next_batch_targets, axis=0)
            next_batch_inputs.clear()
            next_batch_targets.clear()
    yield np.stack(next_batch_inputs, axis=0), np.stack(next_batch_targets, axis=0)


def intersection_over_union(predicted_classes, real_classes, nr_classes):
    print("intersection_over_union")

    # convert from 1-hot-encoding to category indices
    predicted_classes = predicted_classes.argmax(axis=3)

    # calculate IOU for each class
    result = []
    for cls in range(nr_classes):
        intersection = np.zeros(shape=predicted_classes.shape)
        intersection[np.logical_and(np.equal(predicted_classes, cls), np.equal(real_classes, cls))] = 1
        union = np.zeros(shape=predicted_classes.shape)
        union[np.logical_or(np.equal(predicted_classes, cls), np.equal(real_classes, cls))] = 1
        result.append(intersection.sum() / union.sum())
    return result


# Daten liegen im data Ordner, der sich im roo des Projektes befindet.
def main():

    # task 1
    #print("Gude!")
    #image, sx, sy, sz = load_image(LIVER_PATH + TRAINING + '/liver_7.nii.gz')
    #segmentation, _, _, _ = load_image(LIVER_PATH + LABELS + '/liver_7.nii.gz')
    #show_image_collage(image, 5, sx, sy, sz)
    #show_image_collage(image, 5, sx, sy, sz, segmentation)
    #image, sx, sy, sz = load_image(LIVER_PATH + TEST + '/liver_132.nii.gz')
    #show_image_collage(image, 5, sx, sy, sz)
    #all_files = load_dataset(LIVER_PATH)
    #print(repr(all_files))

    # task 2
    unet = make_nnunet(
        patch_size=options_liver[0]["patch_size"],
        pool_op_kernel_sizes=options_liver[0]["pool_op_kernel_sizes"],
        conv_kernel_sizes=options_liver[0]["conv_kernel_sizes"],
        nr_classes=3
    )
    plot_model(unet, show_shapes=True, expand_nested=True, to_file='model.png')

    print("Input shape: {}".format(unet.input_shape))
    print("Output shape: {}".format(unet.output_shape))

    #return      # stop here, I don't have enough RAM

    weights_file = os.path.join(CACHE_PATH, "liver-weights")
    if os.path.exists(weights_file):
        print("Model is already trained, loading weights from {}.".format(weights_file))
        unet.load_weights(weights_file)
    else:
        batch_size = 1  # limited by memory

        # select 10 random images for testing
        files, _ = load_dataset(LIVER_PATH)
        file_identifiers = list(files.keys())
        shuffle(file_identifiers)
        training_identifiers = file_identifiers[10:]
        testing_identifiers = file_identifiers[:10]
        files_train = {i: files[i] for i in training_identifiers}
        files_test = {i: files[i] for i in testing_identifiers}

        print("Preprocessing...")
        # count number of training examples and warm up the cache of preprocessed images
        nr_training_examples = 0
        nr_test_examples = 0
        for i, t in make_train_examples(files_train, 0, unet.input_shape[1:4], unet.output_shape[1:4], options_liver[0]["current_spacing"]):
            nr_training_examples += 1
        for i, t in make_train_examples(files_test, 0, unet.input_shape[1:4], unet.output_shape[1:4], options_liver[0]["current_spacing"]):
            nr_test_examples += 1

        print("Training!")
        batches_per_epoch = math.ceil(nr_training_examples / batch_size)
        batches_per_test_run = math.ceil(nr_test_examples / batch_size)
        # train!
        for epoch in range(25):
            print("Epoch {} of 25".format(epoch))
            batches = make_train_batches(files_train, 0, unet.input_shape[1:4], unet.output_shape[1:4], batch_size, options_liver[0]["current_spacing"])
            for batch, (batch_xs, batch_ys) in enumerate(batches):
                print("Batch {} of {}".format(batch, batches_per_epoch))
                unet.train_on_batch(batch_xs, batch_ys)
            losses = np.zeros(batches_per_test_run)
            batches = make_train_batches(files_test, 0, unet.input_shape[1:4], unet.output_shape[1:4], batch_size, options_liver[0]["current_spacing"])
            for batch, (batch_xs, batch_ys) in enumerate(batches):
                print("Test batch {} of {}".format(batch, batches_per_test_run))
                loss = unet.test_on_batch(batch_xs, batch_ys)
                losses[batch] = loss
            print("Test loss: {}".format(losses.mean()))

        unet.save_weights(weights_file)

main()
