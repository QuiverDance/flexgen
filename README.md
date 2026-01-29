# FlexLLMGen for Llama 3 

----------

## Content
- [Installation](#installation)
- [Usage and Examples](#usage-and-examples)
  - [Get Started with a Single GPU](#get-started-with-a-single-gpu)
- [How It Works](#how-it-works)

## Installation
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### From source
```
git clone https://github.com/QuiverDance/flexgen.git
cd flexgen
pip install -e .
```

## Usage and Examples

### Get Started with a Single GPU

#### LLama-3.1-8B
```
python3 -m flexllmgen.flex_llama --model meta-llama/Llama-3.1-8B --gpu-batch-size 16 --percent 100 0 100 0 100 0
```

#### Llama-3.1-70B
```
python3 -m flexllmgen.flex_llama --model meta-llama/Llama-3.1-70B --gpu-batch-size 16 --percent 0 100 100 0 100 0
```

## How It Works
### Scheduling Overview
<img src="https://github.com/FMInference/FlexLLMGen/raw/main/docs/block_schedule.jpg" alt="image" width="500"></img>

In this figure, Flexgen use Zig-zag block scheduling.
Executes column-by-column for num_gpu_batches instead of full column-by-column scheduling.

### Generation loop mechanism (multi batch)

#### Prologue
prepare as many weights for the first layer as num_gpu_batches and load hidden for the first layer and the first batch.

#### Generate
<img src="https://github.com/FMInference/FlexLLMGen/raw/main/docs/generation_loop_overlap_multi_batch_mechanism.png" alt="image" width="750"></img>

#### Epilogue
store the hidden of the last batch of the layer layer.

### How to store weight
Weights are stored with layer granularity. (e.g. assign n% of the tensors in a layer to the GPU)

### How to store KV cache and activations
KV caches and activations are stored with tensor granularity. (e.g. assign n% of the elements in a tensor to the GPU)

More technical details see our [paper](https://arxiv.org/abs/2303.06865).