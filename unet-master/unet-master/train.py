import numpy as np
from  tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import os
from unet_mini import unet_mini
from mobilnet_unet import mobilenet_unet
from callbacks import ExponentDecayScheduler,LossHistory
from keras import backend as K
from keras import backend 
from data_loader import UnetDataset
#--------------------------------------
# 交叉熵损失函数 cls_weights 类别的权重
#-------------------------------------
def CE(cls_weights):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE
def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def _f_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())

        tp = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = backend.sum(y_pred         , axis=[0,1,2]) - tp
        fn = backend.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score
    return _f_score

def train():
    #-------------------------
    # 细胞图像 分为细胞壁 和其他
    # 初始化 参数
    #-------------------------
    num_classes  = 2 

    input_shape = (512,512,3)
    # 从第几个epoch 继续训练
    
    batch_size = 4

    learn_rate  = 1e-4

    start_epoch = 0
    end_epoch = 100
    num_workers = 4

    dataset_path = 'Medical_Datasets'

    model = mobilenet_unet(num_classes,input_shape=input_shape)

    model.summary()

    # 读取数据图片的路劲
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                        monitor = 'loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.96, verbose = 1)
    early_stopping  = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory('logs/', val_loss_flag = False)

    epoch_step      = len(train_lines) // batch_size
    cls_weights     = np.ones([num_classes], np.float32)
    loss = CE(cls_weights)
    model.compile(loss = loss,
                optimizer = Adam(lr=learn_rate),
                metrics = [f_score()])

    train_dataloader    = UnetDataset(train_lines, input_shape[:2], batch_size, num_classes, True, dataset_path)
    
    
    print('Train on {} samples, with batch size {}.'.format(len(train_lines), batch_size))
    model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            # use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, early_stopping,reduce_lr,loss_history]
        )



if __name__=="__main__":
    train()

