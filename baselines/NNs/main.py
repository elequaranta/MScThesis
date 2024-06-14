import wandb
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import tensorflow
from Functions import preprocess_data, get_model
from Configuration import Config


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = preprocess_data(train_size=0.7, val_test_split=0.5, label_columns = [2, 3, 4, 5, 6, 7, 8], lyrics_column=[9])
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    my_model, best_hps = get_model(X_train, X_val, y_train, y_val)
    conf = Config().get_config()
    try:
        for key, val in best_hps.values.items():
            conf.key = val
    except:
        pass
    wandb.init(project=conf.model_name+'s', name=conf.run_name, config=conf)
    batch_size = conf.batch_size
    def scheduler(epoch, lr):
        if epoch == 1:
            return lr
        else:
            return lr * tensorflow.math.exp(-0.05)
    sched_callback = LearningRateScheduler(scheduler)
    checkpoint_callback = ModelCheckpoint("checkpoints/model-{epoch:02d}ep.keras", monitor="val_categorical_accuracy", verbose=1, save_best_only=False, mode='max')
    logger_callback = WandbMetricsLogger(log_freq='epoch')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=conf.early_patience, restore_best_weights=True)
    print('...Fitting model...')
    results = my_model.fit(X_train, y_train, batch_size=batch_size, epochs=conf.epochs, validation_data=(X_val, y_val), initial_epoch=conf.initial_epoch, callbacks=[checkpoint_callback, logger_callback, early_stopping_callback, sched_callback])
    print('Results ready!')
    for key, value in results.history.items():
        print({key: value})

    wandb.finish()


if __name__ == "__main__":
    main()