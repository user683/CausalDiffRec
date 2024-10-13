# CausalDiffRec

- Paper link: [Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation](https://arxiv.org/pdf/2408.00490)
- Model framework
![png](https://github.com/user683/CausalDiffRec/blob/master/model.png)

## Requirements

- torch==2.1.1+cu121  
- torch_geometric==2.5.3  
- torchaudio==2.1.1+cu121  
- torchvision==0.16.1+cu121  
- tornado==6.4.1  
- dgl==2.0.0+cu121
 

## Run and reproduce

Run following python code (available dataset: "Yelp2018", "Douban") with default hyperparameters to reproduce our results.

```
python train.py --dataset yelp2018 
```
```
python train.py --dataset douban 
```

## Dataset


|  Dataset   |  #Users  |  #Items  |  #Interactions  |   Density   |
|:----------:|:--------:|:--------:|:---------------:|:-----------:|
|    Food    |  7,809   |  6,309   |     216,407     | 4.4 × 10⁻³  |
|  KuaiRec   |  7,175   |  10,611  |    1,153,797    | 1.5 × 10⁻³  |
|  Yelp2018  |  8,090   |  13,878  |     398,216     | 3.5 × 10⁻³  |
|   Douban   |  8,735   |  13,143  |     354,933     | 3.1 × 10⁻³  |


We retain only those users with at least 15 interactions on 
the Food dataset, at least 25 interactions on the Yelp2018 and
Douban datasets, and items with at least 50 interactions on 
these datasets. For all three datasets, only interactions 
with ratings of 4 or higher are considered positive samples. 
For the KuaiRec dataset, interactions with a watch ratio 
of 2 or higher are considered positive samples.

## Acknowledgements

We are particularly grateful to the authors of [DiffRec](https://github.com/YiyanXu/DiffRec), [Graphood-EERM](https://github.com/qitianwu/GraphOOD-EERM), 
and [SELFRec](https://github.com/Coder-Yu/SELFRec) as parts of our code implementation were derived from their work. 
We have cited the relevant references in our paper.

## Reference
```

```
