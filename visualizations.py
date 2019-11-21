from skimage import data, img_as_float
from skimage import exposure
import matplotlib.pyplot as plt

def image_check(x_train):
    w = 25
    h = 25
    fig = plt.figure(figsize=(15, 15))
    columns = 3
    rows = 3
    ax = []
    for i in range(columns*rows):
        img = x_train[i,:,:,:]

        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[i].set_axis_off()
    #     ax[-1].set_title("ax:"+str(i))  # set title
        plt.imshow(img)
    plt.show();










def plot_history_accuracy(model_history):
    history = model_history.history
    loss_values = history['loss']
    val_loss_values = history['val_loss']
    acc_values = history['accuracy']
    val_acc_values = history['val_accuracy']
    topk_acc_values = history['top_k_categorical_accuracy']
    topk_val_values = history['val_top_k_categorical_accuracy']

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(20, 8))
    plt.subplot(131)
    plt.plot(epochs, loss_values, 'g.', label='Training loss')
    plt.plot(epochs, val_loss_values, 'g', label='Validation loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(epochs, acc_values, 'r.', label='Training acc')
    plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(133)
    plt.plot(epochs, topk_acc_values, 'b.', label='Top 5 Categorical Training acc')
    plt.plot(epochs, topk_val_values, 'b', label='Top 5 Categorical Validation acc')
    plt.title('Training and Validation Top 5 Categorical Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show();












def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
       Code is retrieved from scikitimage :
       https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    """
#     image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray);
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
