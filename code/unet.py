import keras
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, Cropping3D, Concatenate
from keras.models import Model
from keras.optimizers import Adam


def make_conv_relu(filters):
    return Conv3D(filters=filters, kernel_size=3, activation=keras.activations.relu)


def make_max_pooling():
    return MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_last")


def make_upconv(filters):
    return Conv3DTranspose(
        filters=filters,
        kernel_size=(2, 2, 2),
        strides=(2, 2, 2),
        data_format="channels_last",
    )


def make_crop(cropping):
    return Cropping3D(cropping=cropping)


def make_concat():
    return Concatenate(axis=4)  # assuming 'channels_last' image format


def make_final_convolution(filters):
    return Conv3D(filters=filters, kernel_size=1, activation=keras.activations.softmax)


def create_unet_3d(nr_classes):
    input_0 = Input(shape=(572, 572, 572, 1))
    conv_1 = make_conv_relu(64)
    layer_1 = conv_1(input_0)
    conv_2 = make_conv_relu(64)
    layer_2 = conv_2(layer_1)

    pool_3 = make_max_pooling()
    layer_3 = pool_3(layer_2)
    conv_4 = make_conv_relu(128)
    layer_4 = conv_4(layer_3)
    conv_5 = make_conv_relu(128)
    layer_5 = conv_5(layer_4)

    pool_6 = make_max_pooling()
    layer_6 = pool_6(layer_5)
    conv_7 = make_conv_relu(256)
    layer_7 = conv_7(layer_6)
    conv_8 = make_conv_relu(256)
    layer_8 = conv_8(layer_7)

    pool_9 = make_max_pooling()
    layer_9 = pool_9(layer_8)
    conv_10 = make_conv_relu(512)
    layer_10 = conv_10(layer_9)
    conv_11 = make_conv_relu(512)
    layer_11 = conv_11(layer_10)

    pool_12 = make_max_pooling()
    layer_12 = pool_12(layer_11)
    conv_13 = make_conv_relu(1024)
    layer_13 = conv_13(layer_12)
    conv_14 = make_conv_relu(1024)
    layer_14 = conv_14(layer_13)

    upconv_15 = make_upconv(512)
    crop_15 = make_crop(4)
    concat_15 = make_concat()
    layer_15_1 = crop_15(layer_11)
    layer_15_2 = upconv_15(layer_14)
    layer_15 = concat_15([layer_15_1, layer_15_2])
    conv_16 = make_conv_relu(512)
    layer_16 = conv_16(layer_15)
    conv_17 = make_conv_relu(512)
    layer_17 = conv_17(layer_16)

    upconv_18 = make_upconv(256)
    crop_18 = make_crop(16)
    concat_18 = make_concat()
    layer_18_1 = crop_18(layer_8)
    layer_18_2 = upconv_18(layer_17)
    layer_18 = concat_18([layer_18_1, layer_18_2])
    conv_19 = make_conv_relu(256)
    layer_19 = conv_19(layer_18)
    conv_20 = make_conv_relu(256)
    layer_20 = conv_20(layer_19)

    upconv_21 = make_upconv(128)
    crop_21 = make_crop(40)
    concat_21 = make_concat()
    layer_21_1 = crop_21(layer_5)
    layer_21_2 = upconv_21(layer_20)
    layer_21 = concat_21([layer_21_1, layer_21_2])
    conv_22 = make_conv_relu(128)
    layer_22 = conv_22(layer_21)
    conv_23 = make_conv_relu(128)
    layer_23 = conv_23(layer_22)

    upconv_24 = make_upconv(64)
    crop_24 = make_crop(88)
    concat_24 = make_concat()
    layer_24_1 = crop_24(layer_2)
    layer_24_2 = upconv_24(layer_23)
    layer_24 = concat_24([layer_24_1, layer_24_2])
    conv_25 = make_conv_relu(64)
    layer_25 = conv_25(layer_24)
    conv_26 = make_conv_relu(64)
    layer_26 = conv_26(layer_25)
    conv_27 = make_final_convolution(nr_classes)
    layer_27 = conv_27(layer_26)

    m = Model(inputs=input_0, outputs=layer_27)
    m.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=Adam()
    )
    return m


def create_unet_3d_small(nr_classes):
    """
    A smaller version of UNet, for which the images better fit into my RAM
    """
    input_0 = Input(shape=(284, 284, 284, 1))
    conv_4 = make_conv_relu(64)
    layer_4 = conv_4(input_0)
    conv_5 = make_conv_relu(64)
    layer_5 = conv_5(layer_4)

    pool_6 = make_max_pooling()
    layer_6 = pool_6(layer_5)
    conv_7 = make_conv_relu(128)
    layer_7 = conv_7(layer_6)
    conv_8 = make_conv_relu(128)
    layer_8 = conv_8(layer_7)

    pool_9 = make_max_pooling()
    layer_9 = pool_9(layer_8)
    conv_10 = make_conv_relu(256)
    layer_10 = conv_10(layer_9)
    conv_11 = make_conv_relu(256)
    layer_11 = conv_11(layer_10)

    pool_12 = make_max_pooling()
    layer_12 = pool_12(layer_11)
    conv_13 = make_conv_relu(512)
    layer_13 = conv_13(layer_12)
    conv_14 = make_conv_relu(512)
    layer_14 = conv_14(layer_13)

    upconv_15 = make_upconv(256)
    crop_15 = make_crop(4)
    concat_15 = make_concat()
    layer_15_1 = crop_15(layer_11)
    layer_15_2 = upconv_15(layer_14)
    layer_15 = concat_15([layer_15_1, layer_15_2])
    conv_16 = make_conv_relu(256)
    layer_16 = conv_16(layer_15)
    conv_17 = make_conv_relu(256)
    layer_17 = conv_17(layer_16)

    upconv_18 = make_upconv(128)
    crop_18 = make_crop(16)
    concat_18 = make_concat()
    layer_18_1 = crop_18(layer_8)
    layer_18_2 = upconv_18(layer_17)
    layer_18 = concat_18([layer_18_1, layer_18_2])
    conv_19 = make_conv_relu(128)
    layer_19 = conv_19(layer_18)
    conv_20 = make_conv_relu(128)
    layer_20 = conv_20(layer_19)

    upconv_21 = make_upconv(64)
    crop_21 = make_crop(40)
    concat_21 = make_concat()
    layer_21_1 = crop_21(layer_5)
    layer_21_2 = upconv_21(layer_20)
    layer_21 = concat_21([layer_21_1, layer_21_2])
    conv_22 = make_conv_relu(64)
    layer_22 = conv_22(layer_21)
    conv_23 = make_conv_relu(64)
    layer_23 = conv_23(layer_22)
    conv_27 = make_final_convolution(nr_classes)
    layer_27 = conv_27(layer_23)

    m = Model(inputs=input_0, outputs=layer_27)
    m.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=Adam()
    )
    return m

def create_unet_3d_small_small(nr_classes):
    """
    An even smaller version of UNet, for which the images better fit into my RAM
    """
    input_0 = Input(shape=(140, 140, 140, 1))
    conv_7 = make_conv_relu(64)
    layer_7 = conv_7(input_0)
    conv_8 = make_conv_relu(64)
    layer_8 = conv_8(layer_7)

    pool_9 = make_max_pooling()
    layer_9 = pool_9(layer_8)
    conv_10 = make_conv_relu(128)
    layer_10 = conv_10(layer_9)
    conv_11 = make_conv_relu(128)
    layer_11 = conv_11(layer_10)

    pool_12 = make_max_pooling()
    layer_12 = pool_12(layer_11)
    conv_13 = make_conv_relu(256)
    layer_13 = conv_13(layer_12)
    conv_14 = make_conv_relu(256)
    layer_14 = conv_14(layer_13)

    upconv_15 = make_upconv(128)
    crop_15 = make_crop(4)
    concat_15 = make_concat()
    layer_15_1 = crop_15(layer_11)
    layer_15_2 = upconv_15(layer_14)
    layer_15 = concat_15([layer_15_1, layer_15_2])
    conv_16 = make_conv_relu(128)
    layer_16 = conv_16(layer_15)
    conv_17 = make_conv_relu(128)
    layer_17 = conv_17(layer_16)

    upconv_18 = make_upconv(64)
    crop_18 = make_crop(16)
    concat_18 = make_concat()
    layer_18_1 = crop_18(layer_8)
    layer_18_2 = upconv_18(layer_17)
    layer_18 = concat_18([layer_18_1, layer_18_2])
    conv_19 = make_conv_relu(64)
    layer_19 = conv_19(layer_18)
    conv_20 = make_conv_relu(64)
    layer_20 = conv_20(layer_19)
    conv_27 = make_final_convolution(nr_classes)
    layer_27 = conv_27(layer_20)

    m = Model(inputs=input_0, outputs=layer_27)
    m.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=Adam()
    )
    return m