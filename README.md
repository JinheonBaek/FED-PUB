# Personalized Subgraph Federated Learning

Official Code Repository for the paper - Personalized Subgraph Federated Learning (ICML 2023): https://arxiv.org/abs/2206.10206.

## Requirement
- Python 3.9.16
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0
- METIS (for data generation), https://github.com/james77777778/metis_python

## Data Generation
Following command lines automatically generate the dataset.
```sh
$ cd data/generators
$ python disjoint.py
$ python overlapping.py
```

## Run 
Following command lines run the experiments for both FedAvg and our FED-PUB.
```sh
$ sh ./scripts/disjoint.sh [gpus] [num_workers]
$ sh ./scripts/overlapping.sh [gpus] [num_workers]
```

- `gpus`: specify gpus to use
- `num workers`: specify the number of workers on gpus (e.g. if your experiment uses 10 clients for every round then use less than or equal to 10 workers). The actual number of workers will be `num_workers` + 1 (one additional worker for a server).

Example
```sh
$ sh ./scripts/disjoint.sh 0,1 10
$ sh ./scripts/overlapping.sh 0,1 10
```

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work. </br>

```BibTex
@article{baek2022personalized,
  title={Personalized subgraph federated learning},
  author={Baek, Jinheon and Jeong, Wonyong and Jin, Jiongdao and Yoon, Jaehong and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2206.10206},
  year={2023}
}
```