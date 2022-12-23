from keras.models import *
from keras.layers import *
import keras.backend as K
import keras
from tensorflow.python.keras.backend import shape

IMAGE_ORDERING =  "channels_last"# channel last
def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
   
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad',
                      data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(filters, kernel, data_format=IMAGE_ORDERING,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING,
                      name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), data_format=IMAGE_ORDERING,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING,
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def get_mobilnet_eocoder(input_shape=(224,224,3),weights_path=""):

    # 必须是32 的倍数
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    alpha = 1.0
    depth_multiplier = 1

    img_input = Input(shape=input_shape)
    #(None, 224, 224, 3) ->(None, 112, 112, 64)
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x
 
    #(None, 112, 112, 64) -> (None, 56, 56, 128)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x
   #(None, 56, 56, 128) -> (None, 28, 28, 256)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x
    # (None, 28, 28, 256) ->  (None, 14, 14, 512)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x
    # (None, 14, 14, 512) -> (None, 7, 7, 1024)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x
    # 加载预训练模型
    if weights_path!="":
        Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)
    # f1: (None, 112, 112, 64)
    # f2: (None, 56, 56, 128)
    # f3: (None, 28, 28, 256)
    # f4: (None, 14, 14, 512)
    # f5: (None, 7, 7, 1024)
    return img_input, [f1, f2, f3, f4, f5]


def mobilenet_unet(num_classes=2,input_shape=(224,224,3)):
    
    #encoder 
    img_input,levels = get_mobilnet_eocoder(input_shape=input_shape,weights_path="model_data\mobilenet_1_0_224_tf_no_top.h5")

    [f1, f2, f3, f4, f5] = levels

    # f1: (None, 112, 112, 64)
    # f2: (None, 56, 56, 128)
    # f3: (None, 28, 28, 256)
    # f4: (None, 14, 14, 512)
    # f5: (None, 7, 7, 1024)

    #decoder
    #(None, 14, 14, 512) - > (None, 14, 14, 512)
    o = f4
    o = ZeroPadding2D()(o)
    o = Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)

    #(None, 14, 14, 512) ->(None,28,28,256)
    o = UpSampling2D(2)(o)
    o = Concatenate(axis=-1)([o,f3])
    o = ZeroPadding2D()(o)
    o = Conv2D(256, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    # None,28,28,256)->(None,56,56,128)
    o = UpSampling2D(2)(o)
    o = Concatenate(axis=-1)([o,f2])
    o = ZeroPadding2D()(o)
    o = Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    #(None,56,56,128) ->(None,112,112,64)
    o = UpSampling2D(2)(o)
    o = Concatenate(axis=-1)([o,f1])
    o = ZeroPadding2D()(o)
    o = Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    #(None,112,112,64) -> (None,112,112,num_classes)

    # 再上采样 让输入和出处图片大小一致
    o = UpSampling2D(2)(o)
    o = ZeroPadding2D()(o)
    o = Conv2D(64, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)

    o = Conv2D(num_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return Model(img_input,o)

if __name__=="__main__":
    mobilenet_unet(input_shape=(512,512,3)).summary()

