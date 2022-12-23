
from keras.layers import Input,Conv2D,Dropout,MaxPooling2D,Concatenate,UpSampling2D
from numpy import pad
from keras.models import Model
def unet_mini(n_classes=21,input_shape=(224,224,3)):

    img_input = Input(shape=input_shape)

   
    #------------------------------------------------------
    # #encoder 部分
    #224,224,3 - > 112,112,32
    conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32,(3,3),activation='relu',padding='same')(conv1)
    pool1 = MaxPooling2D((2,2),strides=2)(conv1)


    #112,112,32 -> 56,56,64
    conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
    pool2 = MaxPooling2D((2,2),strides=2)(conv2)


    #56,56,64 -> 56,56,128
    conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128,(3,3),activation='relu',padding='same')(conv3)

    #-------------------------------------------------
    # decoder 部分
    #56,56,128 -> 112,112,64 
    up1 = UpSampling2D(2)(conv3)
    #112,112,64 -> 112,112,64+128
    up1 = Concatenate(axis=-1)([up1,conv2])
    #  #112,112,192 -> 112,112,64
    conv4  = Conv2D(64,(3,3),activation='relu',padding='same')(up1)
    conv4  = Dropout(0.2)(conv4)
    conv4  = Conv2D(64,(3,3),activation='relu',padding='same')(conv4)

    #112,112,64 - >224,224,64
    up2 = UpSampling2D(2)(conv4)
    #224,224,64 -> 224,224,64+32
    up2 = Concatenate(axis=-1)([up2,conv1])
    # 224,224,96 -> 224,224,32
    conv5 =  Conv2D(32,(3,3),activation='relu',padding='same')(up2)
    conv5  = Dropout(0.2)(conv5)
    conv5  = Conv2D(32,(3,3),activation='relu',padding='same')(conv5)
    
    o = Conv2D(n_classes,1,padding='same')(conv5)

    return Model(img_input,o,name="unet_mini")

if __name__=="__main__":
    model = unet_mini()
    model.summary()