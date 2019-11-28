from skimage import data, img_as_float
from skimage import exposure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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



def plot_history_log(log):
    """
    Takes the model_log file as an input.
    model_log has to be in a pandas DataFrame object e.g. model_log = pd.read_csv('cnnmodels/baseline_training.csv')
    Returns a plotly plot with training and validation accuracies and losses.
    """
    epoch = log.epoch
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training and Validation Loss",
                        "Training and Validation Accuracy",
                        "Top 5 Predictions Training and Validation Accuracy"))

    fig.add_trace(go.Scatter(x=epoch, y=log.loss,
                             mode='markers',
                             name='Training Loss',
                             marker_color='red'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=epoch, y=log.val_loss,
                             name='Validation Loss'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=epoch, y=log.categorical_accuracy,
                             mode='markers',
                            name='Training Accuracy',
                             marker_color='lightseagreen'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=epoch, y=log.val_categorical_accuracy,
                             name='Validation Accuracy',
                             marker_color='lightseagreen'),
                  row=1, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(title_text="Epoch", row=1, col=col)
    fig.update_layout(template='plotly_white', height=500, width=1450,
                      title_text="Model History")

    fig.show()




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
