import random
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd                 
import numpy as np         
import matplotlib.pyplot as plt        

from model import ShallowLinearNetwork, DeepLinearNetwork     
from imgDataset import ImageDataset 
from plotting import plot_losses, plot_metrics, plot_final

random.seed(17)

def one_hot_decode(x):
    x_dec = torch.zeros((x.shape[0], 10), dtype = torch.float32)
    for i in range(x.shape[0]):
        x_dec[i, x[i]] += 1
    
    return x_dec

def extract_samples(data):
    eval_data = []
    test_data = []
    for i in range(data.shape[0] // 2):
        eval_data.append(data[i])
        test_data.append(data[i + 1])
    
    eval_data = np.stack(eval_data, axis = 0)
    test_data = np.stack(test_data, axis = 0)

    x_val = eval_data[:, 1:].astype(np.float32)
    t_val = eval_data[:, 0].astype(np.int32)

    x_test = test_data[:, 1:].astype(np.float32)
    t_test = test_data[:, 0].astype(np.int32)

    return x_val, t_val, x_test, t_test

def get_ds():
    data_train = pd.read_csv("./dataset/MNIST/mnist_train.csv").to_numpy()        
    data_test = pd.read_csv("./dataset/MNIST/mnist_test.csv").to_numpy()  

    x_train = data_train[:, 1:].astype(np.float32)                  
    t_train = data_train[:, 0].astype(np.int32)                     
    
    x_validation, t_validation, x_test, t_test = extract_samples(data_test)
    
    # NORMALIZZAZIONE
    x_train /= 255.0          
    x_validation /= 255.0                                      
    x_test /= 255.0

    # Costruzione dataset
    train_dataset = ImageDataset(x_train, t_train)  
    val_dataset = ImageDataset(x_validation, t_validation)
    test_dataset = ImageDataset(x_test, t_test)

    # Conversione a DataLoader per renderlo compatibile a Pytorch
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, test_dataloader
   
# TRAIN LOOP
def train_model(epochs, learning_rates, n_epochs_eval):

    # DEVICE
    device = torch.device("cpu")

    # DATASET
    train_ds, val_ds, test_ds = get_ds()

    # MODELS: inizializzazione
    models = [
        DeepLinearNetwork(n_input = 784,
                          layers_dims = [2048, 1024, 512, 256, 128, 256, 512, 1024, 2048],
                          n_output = 10,
                          name = "large").to(device),

        DeepLinearNetwork(n_input = 784,
                          layers_dims = [1024, 512, 256, 128, 256, 512, 1024],
                          n_output = 10,
                          name = "medium").to(device),
        
        DeepLinearNetwork(n_input = 784,
                          layers_dims = [512, 256, 128, 256, 512],
                          n_output = 10,
                          name = "small").to(device)
    ]

    f2_scores_val = []
    f2_scores_test = []

    for i in range(len(models)):
        model = models[i]
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rates[i])
        loss_fn = nn.CrossEntropyLoss().to(device)
        writer = SummaryWriter("Exp")

        # TRAIN LOOP
        step = 0

        train_losses = []
        val_losses = []
        val_stds = []
        val_acc = []
        val_prec = []
        val_rec = []
        val_f2 = []

        for epoch in range(epochs):

            # ATTIVAZIONE MODALITA' ADDESTRAMENTO
            model.train()
            losses = []

            # Passaggio del dataloader in "tqdm" per visualizzare progressi a schermo
            batch_iterator = tqdm(train_ds, desc=f"Epoch {epoch:02d}")

            # Prendi le batch di dati e target e esegui l'Error Back-Prop. Algorithm
            for batch in batch_iterator:
                x = batch["data"].to(device)
                t = batch["target"].to(device)

                # FORWARD PASS
                y = model(x)

                # BACKWARD PASS
                loss = loss_fn(y, one_hot_decode(t))                                                # Loss function

                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})         # Setta la loss per printare a schermo
                writer.add_scalar("train_loss", loss.item(), step)
                writer.flush()

                # ----------------------------------------------- PASSAGGI FONDAMENTALI ------------------------------------------------------ #
                loss.backward()                   # BACKPROPAGATION - passaggio della loss al contrario nella rete                                  
                optimizer.step()                  # STEP DI OTTIMIZZAZIONE - passetto dei parametri lungo il negativo del gradiente della loss
                optimizer.zero_grad()             # CANCELLA GRADIENTI - preparati per la prossima iterazione

                step += 1

                losses.append(loss.item())

            train_losses.append(np.mean(np.array(losses)))

            # VALIDATION
            if epochs % n_epochs_eval == 0:
                out = validate_model(model, val_ds, device)
                val_losses.append(out[0])
                val_stds.append(out[1])
                val_acc.append(out[2])
                val_prec.append(out[3])
                val_rec.append(out[4])
                val_f2.append(out[5])

        out_test = validate_model(model, test_ds, device)

        plot_losses(train_losses, val_losses, out_test[0], model.name)
        plot_metrics(val_acc, val_prec, val_rec, val_f2, out_test[-1], model.name)
        
        f2_scores_val.append(val_f2)
        f2_scores_test.append(out_test[-1])

    plot_final(f2_scores_val, f2_scores_test)

    return None
        


def validate_model(model, ds, device):
    val_loss = nn.CrossEntropyLoss().to(device)
    val_losses = []
    
    model.eval()
    with torch.no_grad():
        accuracy = 0
        true_positives = np.zeros(10, dtype = np.int32)
        all_instances = np.zeros(10, dtype = np.int32)
        all_predictions = np.zeros(10, dtype = np.int32)

        n_total = 0

        for batch in ds:
            x = batch["data"].to(device)
            t = batch["target"].to(device)

            y = model(x)

            val_losses.append(val_loss(y, one_hot_decode(t)).item()) 

            predicted_class = torch.argmax(y)
            all_instances[t] += 1
            all_predictions[predicted_class] += 1
            
            if t == predicted_class:
                accuracy += 1
                true_positives[t] += 1


            n_total += 1
        
        recall = np.mean(true_positives / all_instances)
        precision = np.mean(true_positives / all_predictions)

        val_losses = np.array(val_losses)

        mean_loss = np.mean(val_losses)
        std_loss = np.std(val_losses)

        f2 = 2*(precision*recall) / (precision + recall)

        accuracy /= n_total
        print(f"[VALIDATION] Loss: {mean_loss}, STD: {std_loss} \n")
        print(f"[VALIDATION] Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F2: {f2}")
    
    return mean_loss, std_loss, accuracy, precision, recall, f2

def main():
    train_model(epochs = 20, learning_rates = [1e-4, 5e-4, 1e-3], n_epochs_eval = 1)
    return 0


if __name__ == "__main__":
    main()