# Perceptron Visualizer 🔮

This simple script is a visual demonstration of Rosenblatt's Perceptron Algorithm. I was inspired to write this script upon reading the wonderful novel, _Why Machines Learn_ by Anil Ananthaswamy.

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

> **Note:** I recommend trying:<br> `python3 percepton_visualizer.py -a .1 -e 10 -np 500 -s 872983`
