import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np

# input data
import typing
from keras.engine.saving import load_model
from nibabel import Nifti1Image, Nifti2Image

from main import make_train_examples, load_and_preprocess, get_patches, make_image_slices, intersection_over_union, \
    print_prediction_details


@dataclass
class Settings:
    train_losses_file: str      # loss values during training, extracted from logging output of main.py
    test_losses_file: str       # loss values during testing, extracted from logging output of main.py
    test_files_list_file: str
    test_files_images_base_path: str
    test_files_labels_base_path: str
    model_file: str
    current_spacing: typing.List[float]
    image_out_folder: str


base_path = os.path.join(os.path.dirname(__file__), "..")
settings = Settings(
    train_losses_file=os.path.join(base_path, "liver_final_eval2/training_losses.txt"),
    test_losses_file=os.path.join(base_path, "liver_final_eval2/test_losses.txt"),
    test_files_list_file=os.path.join(base_path, "liver_final_eval2/test_files.txt"),
    test_files_images_base_path=os.path.join(base_path, "data/Task03_Liver/imagesTr"),
    test_files_labels_base_path=os.path.join(base_path, "data/Task03_Liver/labelsTr"),
    model_file=os.path.join(base_path, "liver_final_eval2/liver-model"),
    current_spacing=[2.473119, 1.89831205, 1.89831205],
    image_out_folder=os.path.join(base_path, "liver_final_eval2/outimages"),
)


# read the text files with the loss values
def read_losses(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [float(l) for l in lines]


train_losses = read_losses(settings.train_losses_file)
test_losses = read_losses(settings.test_losses_file)


# plot train and test loss over time
def moving_average(arr, size):
    result = np.zeros(len(arr))
    arr = [arr[0]] * size + arr + [arr[-1]] * size
    for i in range(len(result)):
        slice = arr[i:i+2*size+1]
        result[i] = np.mean(slice)
    return result


batches_per_epoch = len(train_losses) / len(test_losses)
plt.plot(
    np.arange(0, len(train_losses)) / batches_per_epoch,
    moving_average(train_losses, 100),
    label="Train losses (moving average)"
)
plt.plot(
    np.arange(0, len(test_losses)) + 1,
    test_losses,
    label="Test losses"
)
plt.legend()
plt.axes().set_ylim([0, 0.05])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Development of the loss over time during training.")
plt.savefig(os.path.join(settings.image_out_folder, "loss_over_time.png"))
plt.show()

# load the model
model = load_model(settings.model_file)

# get the list of test files
test_files = []
with open(settings.test_files_list_file, "r") as f:
    test_file_names = f.readlines()
test_file_names = [f.strip() for f in test_file_names]
test_files = {
    f: (os.path.join(settings.test_files_images_base_path, f), os.path.join(settings.test_files_labels_base_path, f), )
    for f in test_file_names
}

# render triptychs of input image, real labels, predicted labels
def image_to_file(image, filename, segmask=False):
    plt.figure(figsize=(50, 10))
    if segmask:
        slices = make_image_slices(image, 6, img_min=0, img_max=2)
        plt.imshow(slices, vmin=0, vmax=2)
    else:
        slices = make_image_slices(image, 6)
        plt.imshow(slices)
    plt.savefig(filename)
    plt.close()

ious = []
for name, (file_input, file_target) in test_files.items():
    image_input, image_target, image_size, input_padding_before, target_padding_before = load_and_preprocess(
        name, file_input, file_target, 0, model.input_shape[1:4], settings.current_spacing)
    predicted = np.zeros(image_target.shape + np.array(target_padding_before + [0]))
    patch_positions = list(get_patches(image_size, model.input_shape[1:4], model.input_shape[1:4]))
    for inp_min, inp_max, out_min, out_max in patch_positions:
        # get the actual pixel positions by adding the padding that was applied to the image
        inp_min = [inp_min[i] + input_padding_before[i] for i in range(3)]
        inp_max = [inp_max[i] + input_padding_before[i] for i in range(3)]
        out_min = [out_min[i] + target_padding_before[i] for i in range(3)]
        out_max = [out_max[i] + target_padding_before[i] for i in range(3)]

        input_patch = image_input[(inp_min[0]):(inp_max[0]), (inp_min[1]):(inp_max[1]), (inp_min[2]):(inp_max[2]), :]
        target_patch = image_target[(out_min[0]):(out_max[0]), (out_min[1]):(out_max[1]), (out_min[2]):(out_max[2]), :]
        predicted_patch = model.predict(input_patch[np.newaxis, ...]).reshape(target_patch.shape)
        predicted[(out_min[0]):(out_max[0]), (out_min[1]):(out_max[1]), (out_min[2]):(out_max[2]), :] = predicted_patch

    predicted = predicted[target_padding_before[0]:, target_padding_before[1]:, target_padding_before[2]:, :]

    image_to_file(image_input[:, :, :, 0], os.path.join(settings.image_out_folder, name + "-input.png"))
    image_to_file(predicted.argmax(axis=3), os.path.join(settings.image_out_folder, name + "-predicted.png"), segmask=True)
    image_to_file(image_target.argmax(axis=3), os.path.join(settings.image_out_folder, name + "-target.png"), segmask=True)

    iou = intersection_over_union(predicted, image_target.argmax(axis=3), 3)
    print("IOU = " + repr(iou))
    print_prediction_details(predicted, image_target.argmax(axis=3), 3)
    ious = ious + [iou]

ious = np.array(ious)
print("IOU mean:" + repr(ious.mean(axis=0)))
