# Self Supervised clusterer

Combined IIC, and Moco architectures, with some SimCLR notions, to get state of the art unsupervised clustering while retaining interesting image latent representations in the feature space using contrastive learning.

# Installation

Currently successfully tested on Ubuntu 18.04 and Ubuntu 20.04, with python 3.6 and 3.8

Works for Pytorch versions >= 1.4. Launch following command to install all pd

```
pip3 install -r requirements.txt
```


# Logs

All information is logged to tensorboard. If you activate the neptune flag, you can also make logs to Neptune.ai. Before using neptune as a log and output control tool, you need to create a neptune account and get your developer token. Create a `neptune_token.txt` file and store the token in it.

Create in neptune a folder for your outputs, with a name of your choice, then go to `main.py` and modify from line 129 :

```
if args.offline :
    CONNECTION_MODE = "offline"
    run = neptune.init(project='USERNAME/PROJECT_NAME',# You should add your project name and username here
                   api_token=token,
                   mode=CONNECTION_MODE,
                   )
else :
    run = neptune.init(project='USERNAME/PROJECT_NAME',# You should add your project name and username here
               api_token=token,
               )
```
