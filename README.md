# NC Crate Problem

Worked with implementing custom architecture and transfer learning using Keras to predict keypoint on crate.

## Requirements

Install required libraries and packages using:

```bash
pip install -r requirements.txt
```

## Downloading the dataset

To download the dataset:

```bash
python download_dataset.py
```

## Running the scripts:

There's two scripts one for the custom architecture from scratch and one for the transfer learning. Running the script trains the model, saves the trained model, save png of crate with actual and predicted keypoints and also saves the plots for training loss and MAE.

```bash
python custom_architecture.py
```

```bash
python transfer_learning.py
```

### Notes:
Finally there's an idea.png file which has a flowchart of my idea of how to deal with 3D data (something I have not used before) and 3D_traing.ipynb is a small implementation of it.

You can also go through older committs to checkout old jupyter notebooks where I have tried a bunch of ideas and iterated over model training.
