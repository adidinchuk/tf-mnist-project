# use Matplotlib (don't ask)
import matplotlib.pyplot as plt


def visualize_autoencoding(original_data, decoded_data, digits_to_show=10):
    plt.figure(figsize=(20, 4))
    for i in range(digits_to_show):
        # display original
        sub_plot = plt.subplot(2, digits_to_show, i + 1)
        plt.imshow(original_data[i].reshape(28, 28))
        plt.gray()
        sub_plot.get_xaxis().set_visible(False)
        sub_plot.get_yaxis().set_visible(False)

        # display reconstruction
        sub_plot = plt.subplot(2, digits_to_show, i + 1 + digits_to_show)
        plt.imshow(decoded_data[i].reshape(28, 28))
        plt.gray()
        sub_plot.get_xaxis().set_visible(False)
        sub_plot.get_yaxis().set_visible(False)
    plt.show()

def visualize_loss():
    return true