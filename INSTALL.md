# Installation Guide
Clone repository:

```
git clone git@github.com:mariusbock/hawthorne_har.git
cd hawthorne_har
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n hawthorne_har python==3.10.4
conda activate hawthorne_har
```

Install PyTorch distribution:

```
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
```