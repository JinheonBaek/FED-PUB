python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedavg'\
                --dataset 'Cora' \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients 10\
                --seed 42

python3 main.py --gpu $1\
                --n-workers $2\
                --model 'fedpub'\
                --dataset 'Cora' \
                --mode 'disjoint' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients 10\
                --clsf-mask-one\
                --laye-mask-one\
                --norm-scale 3\
                --seed 42
