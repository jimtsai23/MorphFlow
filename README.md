# MorphFlow
Multiview Regenerative Morphing with Dual Flows ([Project Page](https://jimtsai23.github.io/morphflow/), [Paper](https://arxiv.org/abs/2208.01287)).

## Installation
```
git clone https://github.com/jimtsai23/MorphFlow.git
cd MorphFlow
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent. Please install the correct version for your machine.

## Implementation
Users can use collected datasets or their own images as input. For instance, to morph between Lego and Chair provided in Synthetic-NeRF dataset, users can optimize the explicit representation of each scene with
```bash
$ python run.py --config configs/nerf/lego.py
$ python run.py --config configs/nerf/chair.py
```
Then, to morph from Lego to Chair, the optimization is done with
```bash
$ python run.py --config configs/nerf/lego.py --morph lego,chair,weight
```
where weight specifies blending weight of the source, or Lego in this case.


