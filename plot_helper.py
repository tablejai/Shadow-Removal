import matplotlib.pyplot as plt


def plot_histogram_rgb(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def plot_histogram_1Channel(img, title="", new_fig=False):
    if new_fig:
        plt.figure()
    # density=False would make counts
    plt.title(title)
    plt.hist(np.ravel(img), density=True, bins=256)
    plt.xlim([0, 256])
    plt.ylabel('Probability')
    plt.xlabel('Data')
