# CodeModelParameterEfficientFinetuning

This repository contains source code and data for the paper "An Empirical Study of Parameter-Efficient Fine-Tuning Methods for Pre-trained Code Models".

## Abstract

Pre-trained code models (e.g. CodeBERT and CodeT5) have demonstrated their code intelligence in various software engineering tasks, such as code summarization. And full fine-tuning has become the typical approach to adapting these models to downstream tasks. However, full fine-tuning these large models can be computationally expensive and memory-intensive, particularly when training for multiple tasks. To alleviate this issue, several parameter-efficient fine-tuning methods (e.g. Adapter and LoRA) have been proposed to only train a small number of additional parameters, while keeping the original pre-trained parameters frozen. Although these methods claim superiority over the prior techniques, they seldom make a comprehensive and fair comparison on multiple software engineering tasks. Moreover, besides their potential in reducing fine-tuning costs and maintaining approximate performance, the effectiveness of these methods in low-resource, cross-language, and cross-project scenarios is inadequately studied.

To this end, we first conduct experiments by fine-tuning state-of-the-art code models with these methods on both code understanding tasks and code generation tasks. The results show that, by tuning only 0.5\% additional parameters, these methods may achieve comparable or higher performance than full fine-tuning in code understanding tasks, but they may exhibit slightly weaker performance in code generation tasks. We also investigate the impact of these methods with varying numbers of training samples and find that, a considerable number of samples (e.g. 1000 for clone detection) may be required for them to approximate the performance of full fine-tuning. Our experimental results in cross-language and cross-project scenarios demonstrate that by freezing most pre-trained parameters and tuning only 0.5\% additional parameters, these methods achieve consistent improvements in models' transfer learning ability in comparison to full fine-tuning. Our code and data are available at [https://github.com/anonymous-ase23/CodeModelParameterEfficientFinetuning](https://github.com/anonymous-ase23/CodeModelParameterEfficientFinetuning).

## Tasks and Datasets
The datasets except for project-specific code summarization comes from the [CodeXGLUE repository](https://github.com/microsoft/CodeXGLUE). And in each task, you can find the way to download datasets.

Clone detection (BigCloneBench). A model is tasked with measure the semantic similarity between codes. Two existing datasets are included. One is for binary classification between code and the other is for retrieving semantically similar code given code as the query.

Defect detection (Devign). A model is tasked with identifying whether a body of source code contains defects that may be used to attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack. An existing dataset is included.

Code translation (CodeTrans). A model is tasked with translating the code in one programming language to the code in another one. A dataset between Java and C# is newly created.

Code summarization (CodeSearchNet). A model is given the task to generate natural language comments for a code. Existing datasets are included.

Project-specific code summarization. A model is given the task to generate natural language comments for a code whthin a specific project.

## Python Environment
Python                   3.9

torch                    1.13.1

torchaudio               0.8.0a0+a751e1d

torchtyping              0.1.4

torchvision              0.9.0

tornado                  6.2

tqdm                     4.65.0

transformers             4.29.1

translationstring        1.4

tree-sitter              0.20.1


## How To Run 

1. Change directory to each task folder and download the datasets.
2. Choose the run script for PEFT methods. For instance, in the code clone detection task, you can choose the run-none.sh script for full fine-tuning and choose the run-lora.sh script for LoRA.
3. Modify the hypter-parameters for each PEFT method. For instance, in the code clone detection task, you can modify the attn_bn to set a new LoRA bottleneck and you can modify model_name_or_path to adopt a new pre-trained model.

```
cd clone
bash ./run-none.sh # Full fine-tuning
bash ./run-lora.sh # LoRA
bash ./run-adapter.sh # Adapter fine-tuning
bash ./run-parallel-adapter.sh # Parallel Adapter fine-tuning
bash ./run-MHM.sh # MHM fine-tuning
bash ./run-prefix.sh # Prefix fine-tuning
```

# References
Our datasets come from these works.
```
@article{DBLP:journals/corr/abs-2102-04664,
  author    = {Shuai Lu and
               Daya Guo and
               Shuo Ren and
               Junjie Huang and
               Alexey Svyatkovskiy and
               Ambrosio Blanco and
               Colin B. Clement and
               Dawn Drain and
               Daxin Jiang and
               Duyu Tang and
               Ge Li and
               Lidong Zhou and
               Linjun Shou and
               Long Zhou and
               Michele Tufano and
               Ming Gong and
               Ming Zhou and
               Nan Duan and
               Neel Sundaresan and
               Shao Kun Deng and
               Shengyu Fu and
               Shujie Liu},
  title     = {CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding
               and Generation},
  journal   = {CoRR},
  volume    = {abs/2102.04664},
  year      = {2021}
}
@article{Rui2022MPCos,
  title={Low-Resources Project-Specific Code Summarization},
  author={Rui Xie, Tianxiang Hu, Wei Ye, Shikun Zhang},
  journal={37th {IEEE/ACM} International Conference on Automated Software Engineering,
               {ASE} 2022},
  year={2022}
}
```

#

# LICENSE
Our codes follow MIT License.
