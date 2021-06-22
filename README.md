# Marlo-Xinhua-ChiYu
The final project of XinHua and ChiYu is DRL minecraft learning with Marlo environment.

## Requirement
* All of Requirements are listed in requirement.txt.
There is an offical tutorial of Marlo environment installation: [Tutorial](https://marlo.readthedocs.io/en/latest/installation.html)


* Or follow the guide:
```
conda create python=3.7 --name marlo
conda config --add channels conda-forge
conda activate marlo # or `source activate marlo` depending on your conda version
conda install -c crowdai malmo
pip install -U marlo

# install pytorch according to your cuda version
# e.g. pytoch 1.7.1 && cuda 10.1
#    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

pip install tqdm
pip install opencv-python
pip install tensorboardX
```

## Fix the Bug
Edit the file "$install_path$/marlo/base_env_builder.py", to fix error:
```
NameError: name 'etree' is not defined
```
According to https://github.com/crowdAI/marLo/issues/76
```
# change line 577
# from
vp = etree.Element(ns + "VideoProducer")
# to
vp = ElementTree.Element(ns + "VideoProducer")

# add
if w is None:
  w = ElementTree.Element(ns + "Width")
  vp.append(w)
# after line 579 (w = vp.find(ns + "Width"))

# add
if h is None:
  h = ElementTree.Element(ns + "Height")
  vp.append(h)
# after line 581 (h = vp.find(ns + "Height"))

```

## Before Training or Testing
Start Minecraft Clients:
```
$MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
```

## Training
Our codes are named by following rules : [mothod] (DQN/DDPG) + [train/test] + [environment] (Goal/mazeRunner/cliffWalking)  

You can run our training codes by using a simple command.  
e.g.  
```
python DQN_train_Goal.py
```
weights would be saved to corresponding folder (Goal/mazeRunner/cliffWalking)
## Testing
Weights are updated to google drive.

Please download [weights.zip](https://drive.google.com/file/d/12BNl2e5Dh-lzW6v251Da0xy8S7OWZOiu/view?usp=sharing) and unzip
weights.zip in the same folder of codes.
See the layout:

![image](image/layout.jpg)

After the processes above, You can run our testing codes by using a simple command.

e.g.  
```
python DQN_test_Goal.py
```
