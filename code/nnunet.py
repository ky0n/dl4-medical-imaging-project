import keras
from keras import Input, Model
from keras.layers import Conv3D, LeakyReLU, MaxPooling3D, Conv3DTranspose, Concatenate, UpSampling3D
from keras.optimizers import Adam


def conv_leakyrelu(filters, kernel_size, layer):
    layer = Conv3D(filters=filters, kernel_size=kernel_size, padding="same")(layer)
    layer = LeakyReLU(alpha=0.1)(layer)
    return layer


def convolution_step(filters, kernel_size, layer):
    layer = conv_leakyrelu(filters, kernel_size, layer)
    layer = conv_leakyrelu(filters, kernel_size, layer)
    return layer


def pooling(kernel_size, layer):
    return MaxPooling3D(
        pool_size=kernel_size,
        data_format="channels_last",
        padding="same"
    )(layer)


def upconvolution_and_merge(filters, pool_op_kernel_size, last_layer, intermediate_layer):
    upconv = Conv3DTranspose(
        filters=filters,
        kernel_size=pool_op_kernel_size,
        data_format="channels_last",
        padding="same",
        strides=pool_op_kernel_size,
    )
    concat = Concatenate(axis=4)  # assuming 'channels_last' image format
    return concat([intermediate_layer, upconv(last_layer)])


def final_convolution(filters, layer):
    return Conv3D(
        filters=filters,
        kernel_size=1,
        activation=keras.activations.softmax,
        padding="same",
    )(layer)


def make_nnunet(patch_size, pool_op_kernel_sizes, conv_kernel_sizes, nr_classes):

    input = Input(shape=tuple(patch_size + [1]))

    # downwards path
    last_layer = input
    filters = 64
    intermediate_layers = []
    for conv_kernel_size, pool_op_kernel_size in zip(conv_kernel_sizes, pool_op_kernel_sizes[:-1]):
        last_layer = convolution_step(filters, conv_kernel_size, last_layer)
        intermediate_layers.append(last_layer)
        last_layer = pooling(pool_op_kernel_size, last_layer)
        filters *= 2

    last_layer = convolution_step(filters, conv_kernel_sizes[-1], last_layer)

    # upwards path
    reversed_pool_op_kernel_sizes = list(reversed(pool_op_kernel_sizes))
    reversed_conv_kernel_sizes = list(reversed(conv_kernel_sizes[:-1]))
    reversed_intermediate_layers = list(reversed(intermediate_layers))
    for conv_kernel_size, pool_op_kernel_size, intermediate_layer in zip(reversed_conv_kernel_sizes, reversed_pool_op_kernel_sizes, reversed_intermediate_layers):
        filters = int(filters / 2)
        last_layer = upconvolution_and_merge(filters, pool_op_kernel_size, last_layer, intermediate_layer)
        last_layer = convolution_step(filters, conv_kernel_size, last_layer)

    # output layer
    output = final_convolution(nr_classes, last_layer)

    m = Model(inputs=input, outputs=output)
    m.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=Adam()
    )
    return m