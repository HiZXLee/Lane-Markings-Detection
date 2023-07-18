import time
import logging
from helper import get_df, get_partial_df, get_config
from model import get_unet
import tensorflow as tf
from data_generator import train_generator, valid_generator
from losses import focal_tversky_loss, precision, recall, f1_score
import logging
import os

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

logger = logging.getLogger("culane")

if __name__ == "__main__":

    

    config = get_config()

    img_width = int(config["img_width"])
    img_height = int(config["img_height"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    img_channels = int(config["img_channels"])
    prevCheckpoint = config["prevCheckpoint"]
    full_training = bool(config["full_training"])

    if prevCheckpoint.lower() == "none":
        prevCheckpoint = None
    root = config["root"]

    if not os.path.exists(os.path.join(root, "model")):
        os.makedirs(os.path.join(root, "model"))

    if full_training:
        train_dir, train_dir_gt, valid_dir, valid_dir_gt = get_df(root=root)
    else:
        train_dir, train_dir_gt, valid_dir, valid_dir_gt = get_partial_df(root=root)

    model = get_unet(
        img_height=img_height, img_width=img_width, img_channels=img_channels
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss=focal_tversky_loss,
        metrics=[precision, recall, f1_score],
    )

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        "./model/model_best.h5", verbose=1, save_best_only=True
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss"),
        tf.keras.callbacks.CSVLogger("./model/history.csv", append=True),
        checkpointer,
    ]

    if prevCheckpoint is not None:
        try:
            model.load_weights(prevCheckpoint)
        except Exception as e:
            logger.error(e)

    training_start_time = time.time()

    model.fit(
        train_generator(train_dir=train_dir, train_dir_gt=train_dir_gt),
        steps_per_epoch=len(train_dir) // batch_size,
        epochs=epochs,
        validation_data=valid_generator(valid_dir=valid_dir, valid_dir_gt=valid_dir_gt),
        validation_steps=len(valid_dir) // batch_size,
        callbacks=callbacks,
    )

    training_end_time = time.time()
    model.save("./model/model_last.h5")

    logger.info(
        f"Completed training in {(training_end_time-training_start_time)/60:.2f}min"
    )
