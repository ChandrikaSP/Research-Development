## This GitHub repository has PyTorch code implementation sof both SVG-LP and FutureGAN.
# Stochastic Video Generation with a Learned Prior for anomaly detection
This is code from the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus. It is further updated for our task of anomaly detection for SM-MNIST dataset.

##  Training on Stochastic Moving MNIST (SM-MNIST)
To train the SVG-LP model on the 2 digit SM-MNIST dataset run: 
```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```
If the MNIST dataset doesn't exist, it will be downloaded to the specified path.


To generate images with a pretrained SVG-LP model run:
```
python generate_svg_lp.py --model_path --log_dir /generated/images/will/save/here/

```

# FutureGAN: PyTorch Implemetation

This is the official PyTorch implementation of FutureGAN. The code accompanies the paper ["FutureGAN: Anticipating the Future Frames of Video Sequences using Spatio-Temporal 3d Convolutions in Progressively Growing GANs"](https://arxiv.org/abs/1810.01325).

~~~
__Train the Network__<br>

~~~~
$ python train.py --data_root='<path/to/trainsplit/of/your/dataset>'
~~~~

If you want to display the training progress on Tensorboard, set the `--tb_logging` flag:
~~~~
$ python train.py --data_root='<path/to/trainsplit/of/your/dataset>' --tb_logging=True
~~~~

To resume training from a checkpoint, set `--use_ckpt=True` and specify the paths to the generator `ckpt_path[0]` and discriminator `ckpt_path[1]` like this:
~~~~
$ python train.py --data_root='<path/to/trainsplit/of/your/dataset>' --use_ckpt=True --ckpt_path='<path_to_generator_ckpt>' --ckpt_path='<path_to_discriminator_ckpt>'
~~~~
__Test and Evaluate the Network__<br>
To generate predictions with a trained FutureGAN, use the `--data_root` and `--model_path` flags to specify the path to your test data and generator weights and run:
~~~~
$ python eval.py --data_root='<path/to/testsplit/of/your/dataset>' --model_path='<path_to_generator_ckpt>'
~~~~

~~~~
$ python eval.py --data_root='<path/to/testsplit/of/your/dataset>' --model_path='<path_to_generator_ckpt>' --metrics='mse' --metrics='psnr'
~~~~

