import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(train_losses, val_losses, test_loss, name):

    _, ax = plt.subplots(1, 1)

    ax.plot(range(len(train_losses)), train_losses, label = "Training Loss")
    ax.plot(range(len(val_losses)), val_losses, color = "red", label = "Validation Loss")
    ax.hlines(test_loss, 0, len(train_losses), color = "green", linestyles = "dashed", label = "Test Loss")
    ax.legend()

    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy Loss")

    path_img = "./imgs/MNIST/" + name
    name_img = name + "_loss.png"
                             
    plt.savefig(os.path.join(path_img, name_img))
    plt.close()

    pass


def plot_metrics(accuracy, precision, recall, f2, test_f2, name):
    
    _, ax = plt.subplots(1, 1)

    line1 = ax.plot(range(len(accuracy)), accuracy, color = "blue", label= "Validation Accuracy")
    line2 = ax.plot(range(len(precision)), precision, color = "red", label = "Validation Precision")
    line3 = ax.plot(range(len(recall)), recall, color = "green", label = "Validation Recall")
    line4 = ax.plot(range(len(f2)), f2, color = "black", label = "Validation F2 Score")
    line5 = ax.hlines(test_f2, 0, len(accuracy), color = "black", linestyles = "dashed", label = "Test F2 Score")

    ax.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Metrics")

    path_img = "./imgs/MNIST/" + name
    name_img = name + "_metrics.png"

    plt.savefig(os.path.join(path_img, name_img))
    plt.close()

    pass

def plot_final(f2_scores_val, f2_scores_test):
    
    _, ax = plt.subplots(1, 1)
    colors = ["blue", "red", "green"]
    names = ["Large", "Medium", "Small"]
    for i in range(len(f2_scores_val)):
        i_label = names[i] + "Validation F2"
        t_label = names[i] + "Test F2"
        ax.plot(range(len(f2_scores_val[i])), f2_scores_val[i], color = colors[i], label = i_label)
        ax.hlines(f2_scores_test[i], 0, len(f2_scores_val[i]), color = colors[i], linestyles = "dashed", label = t_label)

    ax.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Metrics")

    plt.savefig(os.path.join("./imgs/MNIST/", "val_F2_scores.png"))
    plt.close()

    pass