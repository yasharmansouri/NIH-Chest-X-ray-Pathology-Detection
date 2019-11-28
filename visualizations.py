from skimage import data, img_as_float
from skimage import exposure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def image_check(y_train):
    fig = plt.figure(figsize=(16,32))
    for i in range(32):
        ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i][:,:,0], cmap='bone')
        ax.set_title(', '.join([label for label, check in zip(all_labels, y_train[i]) if check==1]),
                     fontsize=15)



def plot_history_log(log):
    """
    Takes the model_log file as an input.
    model_log has to be in a pandas DataFrame object e.g. model_log = pd.read_csv('cnnmodels/baseline_training.csv')
    Returns a plotly plot with training and validation accuracies and losses.
    """
    epoch = log.epoch
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Training and Validation Loss",
                        "Training and Validation Categorical Accuracy",
                        "Training and Validation AUC"))

    #training and valid loss
    fig.add_trace(go.Scatter(x=epoch, y=log.loss,
                             mode='markers',
                             name='Training Loss',
                             marker_color='red'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=epoch, y=log.val_loss,
                             name='Validation Loss'),
                  row=1, col=1)
    #training and valid categorical acc
    fig.add_trace(go.Scatter(x=epoch, y=log.categorical_accuracy,
                             mode='markers',
                            name='Training Categorical Accuracy',
                             marker_color='lightseagreen'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=epoch, y=log.val_categorical_accuracy,
                             name='Validation Categorical Accuracy',
                             marker_color='lightseagreen'),
                  row=1, col=2)
    #training and valid auc
    fig.add_trace(go.Scatter(x=epoch, y=log.auc_1,
                             mode='markers',
                            name='Training AUC',
                             marker_color='purple'),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=epoch, y=log.val_auc_1,
                             name='Validation AUC',
                             marker_color='purple'),
                  row=1, col=3)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="AUC", row=1, col=3)

    for col in [1, 2, 3]:
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
