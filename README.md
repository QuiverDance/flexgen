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

### Main CLI Options

---
#### --model (str)
- HF model id.
- Llama example: `meta-llama/Llama-3.1-8B`

#### --prompt-len (int), --gen-len (int)
- prompt 길이 및 생성 길이.

#### --gpu-batch-size (int), --num-gpu-batches (int)
- 한번에 GPU에서 계산하는 batch 크기 / 하나의 block 안에 포함되는 GPU 배치(mirco batch)의 개수
- Zig-zag block scheduling에서 한 레이어의 가중치를 재사용하기 위해 num-gpu-batches개 GPU 배치를 묶어 처리

#### --percent (6 ints)
`--percent W_GPU W_CPU KV_GPU KV_CPU ACT_GPU ACT_CPU`

- Weight / KV cache / Activation(hidden)을 GPU/CPU에 얼마나 둘지 결정.
- Disk 비율은 자동으로 나머지(100 - GPU - CPU)로 계산됨.
- 코드 제약:
  - Activation(ACT)은 현재 구현에서 GPU 100% 또는 CPU 100% 또는 Disk 100%만 지원(혼합 비율 불가).
  - Cache(KV)는 100% GPU/CPU/Disk가 아니면 mixed 경로를 타는데,
    `--compress-cache`를 켠 경우 mixed cache는 지원하지 않음(단일 디바이스만 가능).
    `--cpu-cache-compute`를 켠 경우 cpu에서 어텐션 연산 수행

---

## How It Works
Flexgen은 weight/KV cache/activation을 GPU/CPU/DISK로 분산 배치하고,
추가 CUDA stream을 사용하여 I/O(load/store)와 compute를 overlap하여 실행한다. 

### Scheduling Overview
<img src="https://github.com/FMInference/FlexLLMGen/raw/main/docs/block_schedule.jpg" alt="image" width="500"></img>

Flexgen은 Zig-zag block scheduling을 사용한다.

이때 전체 열 단위(column-by-column)로 실행하지 않고, num_gpu_batches 만큼 열 단위로 실행한다.

block size는 GPU batch size 와 the number of GPU batches의 곱이다.

### Generation loop mechanism (multi batch)

#### 1.Prologue
첫 번째 레이어를 위한 가중치를 num_gpu_batches 만큼 준비하고, 첫 번째 배치, 첫 번째 레이어에서 사용할 hidden을 load 한다.

#### 2.Generate
- i: 생성 단계 index
- j: 레이어 index
- k: GPU batch index

3개의 CUDA stream을 사용.
- load_weight_stream : 다음 레이어 weight prefetch
- load_cache_stream  : 다음 단계 cache read prefetch
- store_cache_stream : 이전 단계의 cache write-back

compute(forward)는 기본 stream에서 실행되고, load/store는 위 stream에서 겹쳐질 수 있다.

<img src="https://github.com/QuiverDance/flexgen/blob/readme/docs/generation_loop_overlap_multi_batch_mechanism.png" alt="image" width="750"></img>

#### 3.Epilogue
마지막 배치, 마지막 레이어에 대한 hidden을 store 한다.

### How to store weight
Weight는 layer granularity로 저장한다. (e.g. assign n% of the tensors in a layer to the GPU)

Weight Tensor 정보(tensor 크기, tensor path 등)가 담긴 weight spec을 tensor 크기 기준으로 누적(cumsum)해서,
각 weight tensor가 전체 중 어느 위치에 있는지를 계산한다. 계산된 위치가 가리키는 device에 weight를 저장한다.

### How to store KV cache and activations
KV caches 와 activations은 tensor granularity로 저장한다. (e.g. assign n% of the elements in a tensor to the GPU)

GPU, CPU, DISK 저장 비율만큼 segment length를 구해 각 device에 나누어 저장한다.

### Differences from base flexgen logic

- GQA attention 추가
  - prefill: llama_gqa
  - decode:: llama_gqa_gen
  - q는 n_head, kv는 num_key_value_heads 사용 

- RoPE, RMSNorm 추가

More technical details see our [paper](https://arxiv.org/abs/2303.06865).