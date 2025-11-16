import tensorflow as tf
import sys
from tensorflow import keras

# Logger για διπλή εκτύπωση: terminal + αρχείο
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")  # Άνοιγμα αρχείου για γράψιμο

    def write(self, message):
        self.terminal.write(message)  # Τυπώνει στο terminal
        self.log.write(message)        # Τυπώνει και στο αρχείο

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        train_acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        print(f"Epoch {epoch+1}: "
              f"Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
              f"Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}")

        
