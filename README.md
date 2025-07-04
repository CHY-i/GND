
# GenerativeDecoder
Training a autoregressive network to do quantum error correction.

Link to the article: https://doi.org/10.48550/arXiv.2503.21374

## Generate Surface Code d=3 k=1:

```python
python code_generator.py --d 3 --k 1 --seed 0 -c_type 'sur'
```
if k> k' , where k' is the number of logical qubits, will remove stabilizers randomly from origin code.

## Training a MADE or TraDE and save the network

### MADE
```python
python training.py -save True -n_type 'made' -c_type 'sur' -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device 'cuda:0' -batch 10000 -epoch 50000 -depth 3 -width 20
```
### TraDE
```python
python training.py -save True -n_type 'trade' -c_type 'sur' -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device 'cuda:0' -batch 10000 -epoch 50000 -d_model 128 -n_heads 4 -d_ff 512 -n_layers 2 
```

```python
python Block_training.py -save True -n_type 'trade' -c_type 'qcc' -n 90 -d 10 -k 8 -seed 0 -er 0.13 -device 'cuda:1' -batch 10000 -epoch 500000 -d_model 256 -n_heads 4 -d_ff 256 -n_layers 3 -dtype 'float32'
```
## Correction

### Loading network and forward to do error correction and save the logical error rate

### depolarized
```python
python forward_decoding.py -save True -c_type 'sur' -n 13 -d 3 -k 1 -seed 0  -device 'cuda:0' -n_type 'made' -e_model 'dep' -trials 10000 -er 0.189
```

