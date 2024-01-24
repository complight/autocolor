# *AutoColor:* Learned Light Power Control for Multi-Color Holograms
[Yicheng Zhan](https://github.com/AlberTgarY),
[Koray Kavaklı](https://www.linkedin.com/in/koray-kavakli-75949241/),
[Hakan Ürey](https://mems.ku.edu.tr/),
[Qi Sun](https://qisun.me/),
and [Kaan Akşit](https://kaanaksit.com)

<img src='./media/schematic.png' width=800>


[\[Website\]](http://complightlab.com/autocolor_/), [\[Manuscript\]](https://arxiv.org/abs/2305.01611), [\[Dataset\]](https://github.com/complight/hologram_dataset/tree/main/diffusion)


## Description
*AutoColor* introduces a light-weight neural network that is able to predict required laser powers for a target scene when reconstructed using multi-color holograms (See [*Multi-color Holograms*](https://github.com/complight/multi_color) for more).
The predicted laser powers could be used as the starting point for *Multi-color Holograms* optimization pipeline.
This way, *Multi-color Holograms* pipeline is able to calculate multi-color hologram in just 70 steps rather than 1000 steps like the original *Multi-color Holograms* pipeline.
Obviously, this improvement with *AutoColor* saves time and computational resources, bringing multi-color optimization times from minutes to a few tens of seconds. 


### Citation
If you find this repository useful for your research, please consider citing our work using the below `BibTeX entry`.
```
@misc{zhan2023autocolor,
    doi = {XXXX},
    url = {https://arxiv.org/abs/YYYY},
    author = {Zhan, Yicheng and Kavaklı, Koray and Urey, Hakan and Sun, Qi and Akşit, Kaan},
    keywords = {ZZZZ},
    title = {*Autocolor:* Learned Light Power Control for Multi-Color Holograms},
    publisher = {arXiv},
    year = {2023},
    copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```


## Getting started
This repository contains a code base for estimating laser powers required for a target scne when reconstructed using multi-color holograms.


### (0) Requirements
Before using this code in this repository, please make sure to have the right dependencies installed.
In order to install the main dependency used in this project, please make sure to use the below syntax in a Unix/Linux shell:

```shell
cd autocolor
pip3 install -r requirements.txt
```

Note that we often update `odak`, if this `requirements.txt` fails, please use the below syntax to install odak:

```shell
pip3 install git+https://github.com/kaanaksit/odak
```


### (1) Runtime
Once you have the main dependency installed, you can run the code base using the default settings by providing the below syntax:

```shell
git clone git@github.com:complight/autocolor.git
```
```shell
cd autocolor
```
```shell
python3 main.py --settings settings/holoeye.txt --weights weights/weights.pt --input ANIMAGEFILE.png
```

Please note that `ANIMAGEFILE.png`in the above syntax must be replaced with a target image.
Target image could be at any resolutions and RGB but we highly recommend to follow the same resolution.
If you are looking into finding sample images, consider visiting our [images repository](https://github.com/complight/images).
In the above example, `holoeye.txt` saves the estimation at `~/output/autocolor/ANIMAGEFILE.pt`.
You can use this pt file that contains laser power estimation with `*Multi-color Holograms*`.
In the `*HOLOHDR*`, repository, we provide a sample setting for this purpose as `settings/autocolor.txt`. 
A sample usage is as follows:

```shell
cd ..
```

```shell
git clone git@github.com:complight/multi_color.git
```

```shell
cd multi_color
```

```shell
python3 main.py --settings settings/autocolor.txt
```

### (2) Reconfiguring the code for training purposes
Please consult the settings file found in `settings/sample.txt`, where you will find a list of self descriptive variables that you can modify according to your needs.
This way, you can create a new settings file or modify the existing one.

If you are willing to use the code with another settings file, please use the following syntax:

```shell
python3 main.py --settings settings/sample.txt
```


## Support
For more support regarding the code base, please use the issues section of this repository to raise issues and questions.
