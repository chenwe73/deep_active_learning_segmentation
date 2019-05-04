import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os


isSeg = True


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()
    
    if (isSeg):
        plt.plot(history["mean_iou"])
        plt.plot(history["val_mean_iou"])
        plt.title("miou")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
    plt.close()

def plot_3d(data):
    shape = np.shape(data)
    iteration = shape[0]
    epoch = shape[1]
    
    x = np.arange(epoch)
    y = np.arange(iteration)
    xs, ys = np.meshgrid(x, y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, data, cmap="coolwarm")
    #ax.set_zlim3d(0.5,0.6)

def get_history(out_path, iteration):
    history_path = out_path + "history/history_"
    history = []
    for i in range(iteration):
        h = pickle.load( open( history_path + str(i), "rb" ) )
        history.append(h)
    return history

def plot_combined(iteration, out_path, query):
    for op in out_path:
        plot(iteration, op, query)
    
    for op in out_path:
        history = get_history(op, iteration)
        converge = []
        for i in range(iteration):
            converge.append(np.amax(history[i]["val_acc"]))
        plt.plot(query, converge, label=op)
        #plt.plot(query, np.ones(np.shape(query)) * np.mean(converge))
    plt.legend(loc="best")
    plt.title("Max Validation Pixel-wise Accuracy")
    plt.ylabel("Max Validation Pixel-wise Accuracy")
    plt.xlabel("Number of Labelled Data")
    plt.savefig("./tmp/val_acc.png")
    plt.close()
    
    if (isSeg):
        for op in out_path:
            history = get_history(op, iteration)
            converge = []
            for i in range(iteration):
                converge.append(np.amax(history[i]["val_mean_iou"]))
            plt.plot(query, converge, label=op)
            #plt.plot(query, np.ones(np.shape(query)) * np.mean(converge))
        plt.legend(loc="best")
        plt.title("Validation Mean IOU")
        plt.ylabel("Max Validation Mean IOU")
        plt.xlabel("Number of Labelled Data")
        plt.savefig("./tmp/val_mean_iou.png")
        plt.close()

def plot(iteration, out_path, query):
    history = get_history(out_path, iteration)
    legend_title = "Labelled Pool"
    xlabel = "Epochs"
    
    for i in range(iteration):
        plt.plot(history[i]["acc"], label=query[i])
    plt.legend(loc="best", title=legend_title)
    plt.title(out_path + " Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.savefig(out_path + "acc.png")
    plt.close()
    
    for i in range(iteration):
        plt.plot(history[i]["val_acc"], label=query[i])
    plt.legend(loc="best", title=legend_title)
    plt.title(out_path + " Validation Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.xlabel(xlabel)
    plt.savefig(out_path + "val_acc.png")
    plt.close()
    
    if (isSeg):
        for i in range(iteration):
            plt.plot(history[i]["mean_iou"], label=query[i])
        plt.legend(loc="best", title=legend_title)
        plt.title(out_path + " Mean IOU")
        plt.ylabel("Mean IOU")
        plt.xlabel(xlabel)
        plt.savefig(out_path + "mean_iou.png")
        plt.close()
        
        for i in range(iteration):
            plt.plot(history[i]["val_mean_iou"], label=query[i])
        plt.legend(loc="best", title=legend_title)
        plt.title(out_path + " Valicdation Mean IOU")
        plt.ylabel("Valicdation Mean IOU")
        plt.xlabel(xlabel)
        plt.savefig(out_path + "val_mean_iou.png")
        plt.close()
    
        surface = []
        for i in range(iteration):
            surface.append(history[i]["val_mean_iou"])
        surface = np.array(surface)
        plot_3d(surface)
        plt.title(out_path + " Validation Mean IOU")
        plt.savefig(out_path + "history_3d.png")
        plt.close()
    
    """
    entropy_path = out_path + "entropy/entropy_"
    entropy = []
    for i in range(iteration):
        e = np.load(entropy_path + str(i) + ".npy")
        entropy_image = np.mean(e, axis=1)
        plt.hist(entropy_image, alpha=1, label=query[i])
        entropy.append(entropy_image)
    plt.legend(loc="best")
    plt.savefig(out_path + "entropy_hist.png")
    plt.close()
    """

def main():
    directory = "./tmp/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    iteration = 1 + 4
    query_batch = 600
    init_batch = 600
    query = np.arange(init_batch, init_batch+query_batch*(iteration), query_batch)
    print(query)
    
    out_path = ["./random/", "./entropy/"]
    plot_combined(iteration, out_path, query)
    
    

main()


