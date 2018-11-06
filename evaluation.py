import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def get_bins(predictions, valid_labels):
    """
    Get validation accuracies for different confidence levels.
    :param predictions:         Prediction for each validation sample.
    :param valid_labels:        Label for each validation sample.
    :return:                    Accuracies of 10 bins and their average confidences.
    """

    confidences = np.zeros(10, dtype=np.float32)
    num_correct = np.zeros(10, dtype=np.int32)
    num_total = np.zeros(10, dtype=np.int32)

    for prediction, label in zip(predictions, valid_labels):

        cls = np.argmax(prediction)
        conf = prediction[cls]

        correct = cls == label

        bin_idx = min(int(np.floor(conf * 10)), 9)

        if correct:
            num_correct[bin_idx] += 1

        confidences[bin_idx] += conf
        num_total[bin_idx] += 1

    bins = np.zeros(10, dtype=np.float32)
    avg_confs = np.zeros(10, dtype=np.float32)

    for i in range(10):

        if num_total[i] == 0:
            bins[i] = 0
            avg_confs[i] = 0
        else:
            bins[i] = num_correct[i] / num_total[i]
            avg_confs[i] = confidences[i] / num_total[i]

    return bins, avg_confs, num_total


def get_ece(bins, avg_confs, num_total):
    """
    Calculate the expected calibration error.
    :param bins:            Accuracy for each confidence bin.
    :param avg_confs:       Average confidence for each bin.
    :return:                Error.
    """

    ece = 0
    total = np.sum(num_total)

    for bin, avg_conf, count in zip(bins, avg_confs, num_total):
        ece += (count / total) * np.abs(bin - avg_conf)

    return ece


def plot_bins(bins, avg_confs):

    ax = plt.gca()
    num_bins = 10
    space = 0.3

    for i in range(num_bins):
        plt.bar(i * space, avg_confs[i], width=space, alpha=0.2, color="r")
        plt.bar(i * space, bins[i], width=space, alpha=0.7, color="b")

    ax.set_xticks([i * space - space / 2 for i in range(num_bins + 1)])
    ax.set_xticklabels([str(i / num_bins) for i in range(num_bins + 1)])

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    plt.show()
