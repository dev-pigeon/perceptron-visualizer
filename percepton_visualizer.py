import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import os
import imageio.v2 as imageio


def get_frames_path(frame_dir, frame_num):
    if not os.path.isdir(frame_dir):
        print("making dir")
        print(f"{frame_dir} does not exist, creating...")
        os.mkdir(frame_dir)

    path = os.path.join(frame_dir, f"frame{frame_num}.png")
    return path


def make_gif(frame_dir):
    images = []
    filenames = sorted(os.listdir(frame_dir))

    for filename in filenames:
        if filename.endswith(".png"):
            path = os.path.join(frame_dir, filename)
            images.append(imageio.imread(path))

    imageio.mimsave("perceptron.gif", images, fps=3)


def visualize_perceptron(b: float, w: np.ndarray, y: np.ndarray, X: np.ndarray, epochs: int, alpha=.1, frame_dir=None):
    frame_num = 0
    for i in range(epochs):
        update_occured = False
        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            label = (np.dot(w, xi) * yi) + (b * yi)
            if label <= 0:
                update_occured = True
                w = w + (alpha * yi * xi)
                b = b + (alpha * yi)

        if not update_occured:
            break
        else:
            if frame_dir is not None:
                plot_hyperplane(b, w, X, False, frame_dir, frame_num)
                frame_num += 1
            else:
                plot_hyperplane(b, w, X)

    return {"weights": w, "bias": b}


def plot_hyperplane(b, w, X, final=False, frame_dir=None, frame_num=None):
    plt.clf()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = -(w[0] / w[1]) * x1_vals - (b / w[1])

    colors = ['orange' if label == 1 else 'green' for label in y]
    plt.title("Perceptron Visualizer")
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.plot(x1_vals, x2_vals, color='purple', label="Decision Boundary")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    plt.legend(loc='upper left')
    plt.grid(False)

    if frame_dir is not None and frame_num is not None:
        # get the path
        path = get_frames_path(frame_dir, frame_num)
        plt.savefig(path)
        pass

    if final:
        plt.show()
    else:
        plt.draw()
        plt.pause(1)


def generate_seed():
    now = datetime.now()
    seed = now.year + now.month + now.day + now.hour + now.minute + now.second
    return seed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--alpha', help="The learning rate of the perceptron, generally a small number between 0 and .1.")
    parser.add_argument(
        '-e', '--epochs', help="The number of epochs, or trials, the perceptron will run before giving up.")
    parser.add_argument('-np', '--num-points',
                        help="The number of points to plot.")
    parser.add_argument('-s', '--seed', help="Seed for the point generation.")
    parser.add_argument(
        '-gif', action="store_true", help="Flag to determine if execution should be saved to gif")

    b = 0
    w = np.array([0, 0])

    generate_seed()

    args = parser.parse_args()
    num_points = int(args.num_points) if args.num_points is not None else 10
    alpha = float(args.alpha) if args.alpha is not None else .05
    epochs = int(args.epochs) if args.epochs is not None else 25
    seed = int(args.seed) if args.seed is not None else generate_seed()
    frame_dir = "./frames" if args.gif else None

    np.random.seed(seed)
    X = np.random.uniform(low=0, high=1000, size=(num_points, 2))
    y = np.array([1 if x2 > x1 + 2 else -1 for x1, x2 in X])

    values = visualize_perceptron(
        b=b, w=w, y=y, X=X, epochs=epochs, alpha=alpha, frame_dir=frame_dir)
    plot_hyperplane(b=values['bias'], w=values['weights'], X=X, final=True)

    if args.gif:
        make_gif(frame_dir)
