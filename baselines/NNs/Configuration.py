from keras.losses import BinaryCrossentropy
from keras.metrics import CategoricalAccuracy
import wandb

class Config():
    def __init__(self):
        #Mode settings
        self.mode = 'scratch' #'load'
        self.checkpoint_path = None #'model-01ep.keras'
        self.initial_epoch = int(self.checkpoint_path.split('-')[1].split('ep')[0]) if self.checkpoint_path is not None and self.mode=='load' else 0
        self.model_name = 'CRNN' #'CNN', 'biLSTM'
        self.run_name = 'attempt_1_scheduler_ep2' #for wandb
        #Embeddings/tokenizer
        self.vocab_size = 10000
        self.maxlen = 1500
        self.embedding_dims = 100
        #CNN, CRNN
        self.filters = 128
        self.kernel_size = 10
        self.dropout_rate = 0.15 #CNN only
        #LSTM, CRNN
        self.units = 100
        #General models
        self.batch_size = 32
        self.shuffle = True
        self.embedding_filename = '../GloVe/glove.6B.100d.txt'
        self.data_folder = '../../Data/Music4All-Onion/final_datasets/'
        self.data_filename = 'encoded_df_lyrics.csv'
        self.lr_opt = 0.001
        self.optimizer = 'adam'
        self.steps_per_execution = 200
        #Tuning
        self.tuning = False
        self.search_epochs = 5
        #Training
        self.epochs = 7
        self.loss = BinaryCrossentropy()
        self.metrics = [CategoricalAccuracy()]
        self.early_patience = 5
        
        wandb.config = self

    def get_config(self):
        return self

