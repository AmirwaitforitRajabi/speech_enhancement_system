import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib


def plot_learning_curves(loss, val_loss, path: pathlib.Path):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #plt.axis([1, 40, 0, 0.8])
    plt.ylim([0.01, 0.06])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    #hier muss man die Path angeben, damit es automatisch gespeichert wird
    plt.savefig(path.joinpath('model_loss.png'))


def plot_balken_diagramm(data_sets,lable, x_values = np.array([-5, 0, 5, 10, 15, 20]), delta=True, spacing=0.5,y_lable = 'DNSMOS',x_lable = 'SNRs [dB]'):

    colors = ['black','gold','mediumblue','burlywood','crimson','mediumspringgreen','purple','gold']
    hatches = ['/', '\\','*', 'x', '++','O','+']
    fig, ax = plt.subplots(figsize=(8, 6))
    if delta:
        # Plot each bar plot
        ist = []
        for i, data in enumerate(data_sets):
            if i == 0:
                ist.append(data)
            else:
                sub = []
                for item1, item2 in zip(ist[0], data):
                    sub.append(item2 - item1)
                ist.append(sub)
                offset = (i - len(data_sets) / 2) * spacing
                ax.bar(x_values + offset + 0.5, ist[i], width=spacing, label=lable[i], color=colors[i],
                       hatch=hatches[i])
    else:
        for i, data in enumerate(data_sets):
            offset = (i - len(data_sets) / 2) * spacing
            ax.bar(x_values + offset + 0.5, data, width=spacing, label=lable[i], color=colors[i], hatch=hatches[i])

    ax.set_xticks(x_values + 0.2)
    ax.set_xticklabels([f'{i}' for i in x_values])
    ax.set_xlabel(f'{x_lable}', fontname="Arial", fontsize=45)
    ax.set_ylabel(f'{y_lable}', fontname="Arial", fontsize=45)
    plt.xticks(fontsize=35, fontname="Arial")
    plt.yticks(fontsize=35, fontname="Arial")
    if y_lable == '$\Delta DNSMOS$':
        ax.legend(loc='upper right',fontsize=25)
    else:
        ax.legend(loc='upper left', fontsize=25)
    ax.grid()

    # Show the plot
    plt.show()

def read_snr_data(x):
    x_minus_5 = np.mean(x[0::6])
    x_0 = np.mean(x[1::6])
    x_10 = np.mean(x[2::6])
    x_15 = np.mean(x[3::6])
    x_20 = np.mean(x[4::6])
    x_5 = np.mean(x[5::6])
    return [x_minus_5, x_0, x_5, x_10, x_15, x_20]

