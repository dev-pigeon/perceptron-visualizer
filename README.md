# Perceptron Visualizer 🔮

This simple script is a visual demonstration of Rosenblatt's Perceptron Algorithm operating in 2D space. I was inspired to write this script upon reading the wonderful book _Why Machines Learn_ by Anil Ananthaswamy.

## How It Works 🧠

The **Perceptron algorithm** is one of the earliest machine learning algorithms, invented by American psychologist Frank Rosenblatt in 1957. It finds a linearly separting hyperplane or decision boundary that classifies the into as one of two categories: +1 or -1. The way it accomplishes this is quite elegant, here is the basic idea: <br>

1. Initialize a weight vector **w** and a bias term **b** to zero. <br>
2. For each observation $x_i$ and its corresponding label $y_i \in \{-1, +1\}$ in $X$ do the following: <br>
   2a. If $y_i(w^Tx + b) <= 0$: <br>
   - The weight vector and bias are wrong, update them: <br>
     $w_{new} \rightarrow w_{old} + \alpha*y_i*x_i$ <br>
     $b_{new} \rightarrow b_{old} + \alpha*y_i$
3. Repeat step two for epoch times or until there are no updates.

This script animates this learning process by visualizing how the decision boundary shifts after each iteration of all the points in $X$.

> **Note:** The above algorithm assumes that the bias term is **not** incorporated in the weight vector.

## Requirements ⚙️

- **Unix-based system** (Linux or MacOS)
- **Python 3.x**

## Setup & Run 🚀

Clone the repository: <br>

```bash
git clone https://github.com/dev-pigeon/kafka-vwap-dashboard.git
```

Setup the environment with:

```bash
cd perceptron-visualizer
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Then run:

```bash
python3 perceptron_visualizer.py
```

### Optional Arguments

This script offers four optional arguments:

- `-a`: The learning rate - default = .05
- `-e`: The number of epochs the perceptron will run - default = 25
- `np`: The number of data points to separate - default = 10
- `-s`: The seed for the random point generator (see code for default)

> **Try this:**<br> `python3 percepton_visualizer.py -a .1 -e 10 -np 500 -s 872983`
