# Self Supervised clusterer

Combined IIC, and Moco architectures, with some SimCLR notions, to get state of the art unsupervised clustering while retaining interesting image latent representations in the feature space using contrastive learning.

## Installation

Currently successfully tested on Ubuntu 18.04 and Ubuntu 20.04, with python 3.6 and 3.8

Works for Pytorch versions >= 1.4. Launch following command to install all pd

```
pip3 install -r requirements.txt
```


## Logs

All information is logged to tensorboard. If you activate the neptune flag, you can also make logs to Neptune.ai.

#### Tensorboard

To check logs of your trainings using tensorboard, use the command :

```
tensorboard --logdir=./logs/NAME_OF_TEST/events
```

The `NAME_OF_TEST` is generated automatically for each automatic training you launch, composed of the inputed name of the training you chose (explained further below in commands), and the exact date and time when you launched the training. For example  `test_on_nocadozole_20210518-153531`

#### Neptune

Before using neptune as a log and output control tool, you need to create a neptune account and get your developer token. Create a `neptune_token.txt` file and store the token in it.

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


## Preparing your own data

All datasets will be put in the `./data` folder. As you might have to create various different datasets inside, create a folder inside for each dataset you use, while giving it a linux-friendly name.

To be completed


## Commands

- Adding the `--labels` command means you have ground truth for classes, and you wish to use it in evaluation

- Adding the `--neptune` command means you wish to log your data in neptune (Check logging section)

- `output_k` is the number of clusters

- `model_name` is the name you'll use to keep track of this specific model. Date of training launch will be added to its name.

- `augmentation` is the contrastive loss augmentation types you'll be using. They can be consulted and modified in the `datasets/datasetgetter.py` file.

- `epochs` is the maximal number of epochs you wish to have. It is 1000 by default

- `batch_size` is the training batch size. Default is 32

- `val_batch` is the validation batch size. Default is 10

- `sty_dim` is the size of the style vector. default is 128

- `img_size` size of input images

- `--debug` is a flag for activating debug mode, where the training is very fast, just to check if everything is working fine



###### training from scratch

python main.py --gpu 2  --output_k 9  --model_name=validating_best_image_transfer --augmentation improved_v2 --data_type BBBC021_196  --data_folder ND8D --labels --neptune

###### training using pretrained model

python main.py --gpu 2  --output_k 9  --model_name=validating_best_image_transfer --augmentation improved_v2 --data_type BBBC021_196  --data_folder ND8D --labels --neptune --load_model testing_high_cluster_number_20210604-024131_

###### valiadtion using pretrained model

python main.py --gpu 2  --output_k 9  --model_name=validating_best_image_transfer --augmentation improved_v2 --data_type BBBC021_196  --data_folder ND8D --labels --validation --neptune --load_model testing_high_cluster_number_20210604-024131_
