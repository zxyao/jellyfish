# jellyfish
Repo for 10-615 Art + ML Proj 2

### Prepro
Output a folder of stacked original images + edge images in specified folder. 

Must put the original data set folder inside `examples/hed`.

```bash
cd examples/hed
```
```bash
./prepro.sh data/jellyfish
```

### Run pix2pix model
Modified based on the provided notebook. 

Main modifications:
- Model loading and saving
- Save output imgs

Start from course AMI:
```bash
git clone https://github.com/zxyao/jellyfish
```
```bash
cd jellyfish/pix2pix
```
```bash
source activate tensorflow_p27
```
```bash
python train.py --data jellyfishdata 
```
Rename the wanted checkpoint files
```bash
python test.py --data jellyfishdata
```
