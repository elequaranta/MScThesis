from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Softmax, Conv1D, GlobalMaxPooling1D, Dropout, Concatenate
from keras.models import Model
from Functions import preprocess_data
from Configuration import Config
import tensorflow as tf


config = Config().get_config()
X_train, _, _, _, _, _, _, _ = preprocess_data()
        
        
def build_model(hp): 
    if config.model_name == 'biLSTM':   
        if hp is not None:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            lstm = Bidirectional(LSTM(units=config.units))(inp)
            dense = Dense(7)(lstm)
            out = Softmax()(dense)
            model = Model(raw_inp, out)
            optim_lr = hp.Choice('lr_opt', values=[1e-2, 1e-3, 1e-4])
            optimizer = hp.Choice('optimizer', ['sgd', 'adam', 'adagrad', 'none'])
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = len(X_train)//(hp.Choice('batch_size', values=[32, 64, 128, 256])*config.epochs))
        else:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            lstm = Bidirectional(LSTM(units=config.units))(inp)
            dense = Dense(7)(lstm)
            out = Softmax()(dense)
            model = Model(raw_inp, out)
            optim_lr = config.lr_opt
            optimizer = config.optimizer
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = config.steps_per_execution)
    elif config.model_name == 'CNN':
        if hp is not None:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            conv1 = Conv1D(config.filters,
                            config.kernel_size[0],
                            padding='valid')(inp)
            pooler = GlobalMaxPooling1D(data_format='channels_last')
            pool1 = pooler(conv1)
            conv2 = Conv1D(config.filters,
                            config.kernel_size[1],
                            padding='valid')(inp)
            pool2 = pooler(conv2)
            conv3 = Conv1D(config.filters,
                            config.kernel_size[2],
                            padding='valid')(inp)
            pool3 = pooler(conv3)
            conv_pool = Concatenate()([pool1, pool2, pool3])
            dense1 = Dense(128, activation='relu')(conv_pool)
            drop = Dropout(hp.Float(name='dropout_rate', min_value=0.01, max_value=0.5, sampling='log'))(dense1)
            dense2 = Dense(7, activation='relu')(drop)
            out = Softmax()(dense2)
            model = Model(raw_inp, out)
            optim_lr = hp.Choice('lr_opt', values=[1e-2, 1e-3, 1e-4])
            optimizer = hp.Choice('optimizer', ['sgd', 'adam', 'adagrad', 'none'])
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = len(X_train)//(hp.Choice('batch_size', values=[32, 64, 128, 256])*config.epochs))
        else:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            conv1 = Conv1D(config.filters,
                            config.kernel_size[0],
                            padding='valid')(inp)
            pooler = GlobalMaxPooling1D(data_format='channels_last')
            pool1 = pooler(conv1)
            conv2 = Conv1D(config.filters,
                            config.kernel_size[1],
                            padding='valid')(inp)
            pool2 = pooler(conv2)
            conv3 = Conv1D(config.filters,
                            config.kernel_size[2],
                            padding='valid')(inp)
            pool3 = pooler(conv3)
            conv_pool = Concatenate()([pool1, pool2, pool3])
            dense1 = Dense(128, activation='relu')(conv_pool)
            drop = Dropout(config.dropout_rate)(dense1)
            dense2 = Dense(7, activation='relu')(drop)
            out = Softmax()(dense2)
            model = Model(raw_inp, out)
            optim_lr = config.lr_opt
            optimizer = config.optimizer
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = config.steps_per_execution)
    elif config.model_name=='CRNN':
        if hp is not None:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            conv1 = Conv1D(config.filters,
                            hp.Choice('kernel_size', values=[5, 10, 15]),
                            padding='valid')(inp)
            conv2 = Conv1D(config.filters,
                            hp.Choice('kernel_size', values=[5, 10, 15]),
                            padding='valid')(inp)
            conv3 = Conv1D(config.filters,
                            hp.Choice('kernel_size', values=[5, 10, 15]),
                            padding='valid')(inp)
            conv_pool = Concatenate()([conv1, conv2, conv3])
            lstm = Bidirectional(LSTM(units=config.units))(conv_pool)
            dense = Dense(7)(lstm)
            out = Softmax()(dense)
            model = Model(raw_inp, out)
            optim_lr = hp.Choice('lr_opt', values=[1e-2, 1e-3, 1e-4])
            optimizer = hp.Choice('optimizer', ['sgd', 'adam', 'adagrad', 'none'])
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = len(X_train)//(hp.Choice('batch_size', values=[32, 64, 128, 256])*config.epochs))
        else:
            raw_inp = Input(shape=(config.maxlen,))
            inp = Embedding(config.vocab_size,
                            config.embedding_dims)(raw_inp)
            conv1 = Conv1D(config.filters,
                            config.kernel_size,
                            padding='valid')(inp)
            conv2 = Conv1D(config.filters,
                            config.kernel_size,
                            padding='valid')(inp)
            conv3 = Conv1D(config.filters,
                            config.kernel_size,
                            padding='valid')(inp)
            conv_pool = Concatenate()([conv1, conv2, conv3])
            lstm = Bidirectional(LSTM(units=config.units))(conv_pool)
            dense = Dense(7)(lstm)
            out = Softmax()(dense)
            model = Model(raw_inp, out)
            optim_lr = config.lr_opt
            optimizer = config.optimizer
            if optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=optim_lr)
            elif optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optim_lr)
            elif optimizer == 'adagrad':
                optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=optim_lr)
            else:
                optimizer = None
            model.compile(loss=config.loss, metrics=config.metrics, optimizer=optimizer, steps_per_execution = config.steps_per_execution)
    return model

            