<p align="center">
    <img src="./images/title-llm-as-a-judge.png" width="70%"> <br>
</p>
<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="https://awesome-llm-as-a-judge.github.io/" style="text-decoration: none; font-weight: bold;">🌻 Homepage</a> •
    <a href="#paper-list" style="text-decoration: none; font-weight: bold;"> 📖 Paper List</a> •
    <a href="https://awesome-llm-as-a-judge.github.io/#meta-evaluation" style="text-decoration: none; font-weight: bold;">📊 Meta-eval</a> •
    <a href="https://arxiv.org/abs/2411.15594" style="text-decoration: none; font-weight: bold;">🌟 Arxiv </a> •
    <a href="https://event.baai.ac.cn/activities/878" style="text-decoration: none; font-weight: bold;"> 🔗 Talk </a>
  </p>
</div>


This repo include the papers discussed in our survey paper *[A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594)*

### Reference

Feel free to cite if you find our survey is useful for your research:

```
@article{gu2024surveyllmasajudge,
	title   = {A Survey on LLM-as-a-Judge},
	author  = {Jiawei Gu and Xuhui Jiang and Zhichao Shi and Hexiang Tan and Xuehao Zhai and Chengjin Xu and Wei Li and Yinghan Shen and Shengjie Ma and Honghao Liu and Yuanzhuo Wang and Jian Guo},
	year    = {2024},
	journal = {arXiv preprint arXiv: 2411.15594}
}
```

### 🔔 News  

🔥 [2025-01-28]  We added analysis on **LLM-as-a-Judge** and **o1-like Reasoning Enhancement**, as well as [**meta-evaluation**](https://awesome-llm-as-a-judge.github.io/#meta-evaluation) results on **o1-mini**, **Gemini-2.0-Flash-Thinking-1219**, and **DeepSeek-R1**!  

🌟 [2025-01-16] We shared and discussed the **methodologies**, **applications (Finance, RAG, and Synthetic Data)**, and future research directions of **LLM-as-a-Judge** at BAAI Talk! 🤗  [[**Replay**](https://event.baai.ac.cn/activities/878)]  [[**Methodology**](https://ticket-assets.baai.ac.cn/uploads/%E6%96%B9%E6%B3%95%E8%AE%BA%E3%80%81%E5%BA%94%E7%94%A8%E4%B8%8E%E6%9C%AA%E6%9D%A5%E7%A0%94%E7%A9%B6%E6%96%B9%E5%90%91%E6%8E%A2%E8%AE%A8.pdf)]  [[**RAG & Synthetic Data**](https://ticket-assets.baai.ac.cn/uploads/LLM-as-a-Judge-%E5%BE%90%E9%93%96%E6%99%8B.pdf)]  

🚀 [2024-11-23]  We released [**A Survey on LLM-as-a-Judge**](https://arxiv.org/pdf/2411.15594), exploring **LLMs as reliable, scalable evaluators** and outlining **key challenges and future directions**!  

### Overview of LLM-as-a-Judge 

![overview](./images/paper_structure.png)



### Evaluation Pipelines

![evaluation_pipeline](./images/evaluation_pipeline.jpg)



### Improvement Strategies for LLM-as-a-Judge

![improvement_strategy](./images/improvement_strategy.png)



## Table of Content

[A Survey on LLM-as-a-Judge](#Awesome-LLM-as-a-Judge)

- [Reference](#Reference)
- [Overview of LLM-as-a-Judge](#Overview-of-LLM-as-a-Judge)
- [Evaluation Pipelines of LLM-as-a-Judge](#Evaluation-Pipelines-of-LLM-as-a-Judge)
- [Improvement Strategies for LLM-as-a-Judge](#Improvement-Strategies-for-LLM-as-a-Judge)
- [Table of Content](#Table-of-Content)
- [Paper List](#Paper-List)
  - [1 What is LLM-as-a-Judge?](#1-What-is-LLM-as-a-Judge?)
  - [2 How to use LLM-as-a-Judge?](#2-How-to-use-LLM-as-a-Judge?)
    - [2.1 In-Context Learning](#21-In-Context-Learning)
      - [Generating scores](#Generating-scores)
      - [Solving Yes/No questions](#Solving-Yes/No-questions)
      - [Conducting pairwise comparisons](#Conducting-pairwise-comparisons)
      - [Making multiple-choice selections](#Making-multiple-choice-selections)
    - [2.2 Model Selection](#22-Model-Selection)
      - [General LLM](#General-LLM)
      - [Fine-tuned LLM](#Fine-tuned-LLM)
    - [2.3 Post-processing Method](#23-Post-processing-Method)
      - [Extracting specific tokens](#Extracting-specific-tokens)
      - [Constrained decoding](#Constrained-decoding)
      - [Normalizing the output logits](#Normalizing-the-output-logits)
      - [Selecting sentences](#Selecting-sentences)
    - [2.4 Evaluation Pipeline](#24-Evaluation-Pipeline)
      - [LLM-as-a-Judge for Models](#LLM-as-a-Judge-for-Models)
      - [LLM-as-a-Judge for Data](#LLM-as-a-Judge-for-Data)
      - [LLM-as-a-Judge for Agents](#LLM-as-a-Judge-for-Agents)
      - [LLM-as-a-Judge for Reasoning/Thinking](#LLM-as-a-Judge-for-Reasoning/Thinking)
  - [3 How to improve LLM-as-a-Judge?](#3-How-to-improve-LLM-as-a-Judge?)
    - [3.1 Design Strategy of Evaluation Prompts](#31-Design-Strategy-of-Evaluation-Prompts)
      - [Few-shot promping](#Few-shot-promping)
      - [Evaluation steps decomposition](#Evaluation-steps-decomposition)
      - [Evaluation criteria decomposition](#Evaluation-criteria-decomposition)
      - [Shuffling contents](#Shuffling-contents)
      - [Conversion of evaluation tasks](#Conversion-of-evaluation-tasks)
      - [Constraining outputs in structured formats](#Constraining-outputs-in-structured-formats)
      - [Providing evaluations with explanations](#Providing-evaluations-with-explanations)
    - [3.2 Improvement Strategy of LLMs' Abilities](#32-Improvement-Strategy-of-LLMs'-Abilities)
      - [Fine-tuning via Meta Evaluation Dataset](#Fine-tuning-via-Meta-Evaluation-Dataset)
      - [Iterative Optimization Based on Feedbacks](#Iterative-Optimization-Based-on-Feedbacks)
    - [3.3 Optimization Strategy of Final Results](#33-Optimization-Strategy-of-Final-Results)
      - [Summarize by multiple rounds](#Summarize-by-multiple-rounds)
      - [Vote by multiple LLMs](#Vote-by-multiple-LLMs)
      - [Score smoothing](#Score-smoothing)
      - [Self validation](#Self-validation)
  - [4 How to evaluate LLM-as-a-Judge?](#4-How-to-evaluate-LLM-as-a-Judge?)
    - [4.1 Basic Metric](#41-Basic-Metric)
    - [4.2 Bias](#42-Basic)
      - [Position Bias](#Position-Bias)
      - [Length Bias](#Length-Bias)
      - [Self-Enhancement Bias](#Self-Enhancement-Bias)
      - [Other Bias](#Other-Bias)
    - [4.3 Adversarial Robustness](#43-Adversarial-Robustness)



## Paper List

### 1 What is LLM-as-a-Judge?

### 2 How to use LLM-as-a-Judge? 

#### 2.1 In-Context Learning

##### Generating scores

- **A Multi-Aspect Framework for Counter Narrative Evaluation using Large Language Models** `NAACL` `2024`

  Jaylen Jones, Lingbo Mo, Eric Fosler-Lussier, and Huan Sun. [[Paper](https://aclanthology.org/2024.naacl-short.14)]

- **Generative judge for evaluating alignment.** `ArXiv preprint` `2023`

  Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. [[Paper](https://arxiv.org/abs/2310.05470)]

- **Judgelm: Fine-tuned large language models are scalable judges.** `ArXiv preprint` `2023`

  Lianghui Zhu, Xinggang Wang, and Xinlong Wang. [[Paper](https://arxiv.org/abs/2310.17631)]

- **Large Language Models are Better Reasoners with Self-Verification.** `EMNLP findings` `2023`

  Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. [[Paper](https://aclanthology.org/2023.findings-emnlp.167)]

- **Benchmarking Foundation Models with Language-Model-as-an-Examiner.** `NeurIPS` `2023`

  Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, and Lei Hou. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f64e55d03e2fe61aa4114e49cb654acb-Paper-Datasets_and_Benchmarks.html)]

- **Human-like summarization evaluation with chatgpt.** `ArXiv preprint` `2023`

  Mingqi Gao, Jie Ruan, Renliang Sun, Xunjian Yin, Shiping Yang, and Xiaojun Wan. [[Paper](https://arxiv.org/abs/2304.02554)]

##### Solving Yes/No questions

- **Reflexion: language agents with verbal reinforcement learning.** `NeurIPS` `2023`

  Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. [[Paper](https://papers.nips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.html)]

- **MacGyver: Are Large Language Models Creative Problem Solvers?** `NAACL` `2024`

  Yufei Tian, Abhilasha Ravichander, Lianhui Qin, Ronan Le Bras, Raja Marjieh, Nanyun Peng, Yejin Choi, Thomas Griffiths, and Faeze Brahman. [[Paper](https://aclanthology.org/2024.naacl-long.297)]

- **Think-on-graph: Deep and responsible reasoning of large language model with knowledge graph.** `ArXiv preprint` `2023`

  Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Heung-Yeung Shum, and Jian Guo. [[Paper](https://arxiv.org/abs/2307.07697)]

##### Conducting pairwise comparisons

- **Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.** `NAACL findings` `2024`

  Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. [[Papaer](https://aclanthology.org/2024.findings-naacl.97)]

- **Aligning with human judgement: The role of pairwise preference in large language model evaluators. ** `COLM` `2024`

  Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulic, Anna Korhonen, and Nigel Collier. [[Paper](https://arxiv.org/abs/2403.16950)]

- **LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons using Large Language Models.** `EACL` `2024`

  Adian Liusie, Potsawee Manakul, and Mark Gales. [[Paper](https://aclanthology.org/2024.eacl-long.8)]

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** `NeurIPS` `2023`

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

- **Rrhf: Rank responses to align language models with human feedback without tears.** `ArXiv preprint` `2023`

  Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. [[Paper](https://arxiv.org/abs/2304.05302)]

- **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** `ArXiv preprint` `2023`

  Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. [[Paper](https://arxiv.org/abs/2306.05087)]

- **Human-like summarization evaluation with chatgpt.** `ArXiv preprint` `2023`

  Mingqi Gao, Jie Ruan, Renliang Sun, Xunjian Yin, Shiping Yang, and Xiaojun Wan. [[Paper](https://arxiv.org/abs/2304.02554)]

##### Making multiple-choice selections

- 

#### 2.2 Model Selection

##### General LLM

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** `NeurIPS` `2023`

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

- **AlpacaEval: An Automatic Evaluator of Instruction-following Models.**  `2023`

  Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.  [[Code](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file)]

##### Fine-tuned LLM

- **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** `ArXiv preprint` `2023`

  Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. [[Paper](https://arxiv.org/abs/2306.05087)]

- **Judgelm: Fine-tuned large language models are scalable judges.** `ArXiv preprint` `2023`

  Lianghui Zhu, Xinggang Wang, and Xinlong Wang. [[Paper](https://arxiv.org/abs/2310.17631)]

- **Generative judge for evaluating alignment.** `ArXiv preprint` `2023`

  Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. [[Paper](https://arxiv.org/abs/2310.05470)]

- **Prometheus: Inducing Fine-grained Evaluation Capability in Language Models.** `ArXiv preprint` `2023`

  Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. [[Paper](https://arxiv.org/abs/2310.08491)]

#### 2.3 Post-processing Method

##### Extracting specific tokens

- **xFinder: Robust and Pinpoint Answer Extraction for Large Language Models.** `ArXiv preprint` `2024`

  Qingchen Yu, Zifan Zheng, Shichao Song, Zhiyu Li, Feiyu Xiong, Bo Tang, and Ding Chen. [[Paper](https://arxiv.org/abs/2405.11874)]

- **MacGyver: Are Large Language Models Creative Problem Solvers?** `NAACL` `2024`

  Yufei Tian, Abhilasha Ravichander, Lianhui Qin, Ronan Le Bras, Raja Marjieh, Nanyun Peng, Yejin Choi, Thomas Griffiths, and Faeze Brahman. [[Paper](https://aclanthology.org/2024.naacl-long.297)]

##### Constrained decoding

- **Guiding LLMs the right way: fast, non-invasive constrained generation.** `ICML` `2024`

  Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. [[Paper](https://dl.acm.org/doi/10.5555/3692070.3692216)]

- **XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.** `ArXiv preprint` `2024`

  Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, and Tianqi Chen. [[Paper](https://arxiv.org/abs/2411.15100)]

- **SGLang: Efficient Execution of Structured Language Model Programs.** `NeurIPS` `2025`

  Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)]

##### Normalizing the output logits

- **Reasoning with Language Model is Planning with World Model.** `EMNLP` `2023`

  Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. [[Paper](https://aclanthology.org/2023.emnlp-main.507)]

- **Speculative rag: Enhancing retrieval augmented generation through drafting.** `ArXiv preprint` `2024`

  Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, et al. [[Paper](https://arxiv.org/abs/2407.08223)]

- **Agent-as-a-Judge: Evaluate Agents with Agents. ** `ArXiv preprint` `2024`

  Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. [[Paper](https://arxiv.org/abs/2410.10934)]

##### Selecting sentences

- **Reasoning with Language Model is Planning with World Model.** `EMNLP` `2023`

  Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. [[Paper](https://aclanthology.org/2023.emnlp-main.507)]

#### 2.4 Evaluation Pipeline

##### LLM-as-a-Judge for Models

- **AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback.** `NeurIPS` `2023`

  Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/5fc47800ee5b30b8777fdd30abcaaf3b-Abstract-Conference.html)]

- **Large language models are not fair evaluators.** `ACL` `2024`

  Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. [[Paper](https://aclanthology.org/2024.acl-long.511)]

- **Wider and deeper llm networks are fairer llm evaluators.** `ArXiv preprint` `2023`

  Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. [[Paper](https://arxiv.org/abs/2308.01862)]

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** `NeurIPS` `2023`

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

- **SelFee: Iterative Self-Revising LLM Empowered by Self-Feedback Generation. ** `Blog` `2023`

  Seonghyeon Ye, Yongrae Jo, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, and Minjoon Seo. [[Blog](https://kaistai.github.io/SelFee)]

- **Shepherd: A Critic for Language Model Generation.** `ArXiv preprint` `2023`

  Tianlu Wang, Ping Yu, Xiaoqing Ellen Tan, Sean O’Brien, Ramakanth Pasunuru, Jane Dwivedi-Yu, Olga Golovneva, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. [[Paper](https://arxiv.org/abs/2308.04592)]

- **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** `ArXiv preprint` `2023`

  Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. [[Paper](https://arxiv.org/abs/2306.05087)]

##### LLM-as-a-Judge for Data

- **RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.** `ArXiv preprint` `2023`

  Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. [[Paper](https://arxiv.org/abs/2304.06767)]

- **Rrhf: Rank responses to align language models with human feedback without tears.** `ArXiv preprint` `2023`

  Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. [[Paper](https://arxiv.org/abs/2304.05302)]

- **Stanford Alpaca: An Instruction-following LLaMA model.** `2023`

  Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. [[Code](https://github.com/tatsu-lab/stanford_alpaca)]

- **Languages are rewards: Hindsight finetuning using human feedback.** `ArXiv preprint` `2023`

  Hao Liu, Carmelo Sferrazza, and Pieter Abbeel. [[Paper](https://arxiv.org/abs/2302.02676)]

- **The Wisdom of Hindsight Makes Language Models Better Instruction Followers.** `PMLR` `2023`

  Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, and Joseph E. Gonzalez. [[Paper](https://proceedings.mlr.press/v202/zhang23ab.html)]

- **Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision. ** `NeurIPS` `2023`

  Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David D. Cox, Yiming Yang, and Chuang Gan. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html)]

- **Wizardmath: Empowering mathematical reasoning for large language models via**

  **reinforced evol-instruct. ** `ArXiv preprint` `2023`

  Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. [[Paper](https://arxiv.org/abs/2308.09583)]

- **Self-taught evaluators.** `ArXiv preprint` `2024`

  Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. [[Paper](https://arxiv.org/abs/2408.02666)]

- **Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges.** `ArXiv preprint` `2023`

  Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. [[Paper](https://arxiv.org/abs/2311.03287)]

- **Evaluating Object Hallucination in Large Vision-Language Models.** `EMNLP` `2023`

  Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. [[Paper](https://aclanthology.org/2023.emnlp-main.20/)]

- **Evaluation and analysis of hallucination in large vision-language models.** `ArXiv preprint` `2023`

  Junyang Wang, Yiyang Zhou, Guohai Xu, Pengcheng Shi, Chenlin Zhao, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Jihua Zhu, et al. [[Paper](https://arxiv.org/abs/2308.15126)]

- **Aligning large multimodal models with factually augmented rlhf.** `ArXiv preprint` `2023`

  Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. [[Paper](https://arxiv.org/abs/2309.14525)]

- **MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark.** `ICML` `2024`

  Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. [[Paper](https://openreview.net/forum?id=dbFEFHAD79)]

##### LLM-as-a-Judge for Agents

- **Agent-as-a-Judge: Evaluate Agents with Agents. ** `ArXiv preprint` `2024`

  Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. [[Paper](https://arxiv.org/abs/2410.10934)]

- **Reasoning with Language Model is Planning with World Model.** `EMNLP` `2023`

  Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. [[Paper](https://aclanthology.org/2023.emnlp-main.507)]

- **Reflexion: language agents with verbal reinforcement learning.** `NeurIPS` `2023`

  Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. [[Paper](https://papers.nips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)]

##### LLM-as-a-Judge for Reasoning/Thinking

- **Towards Reasoning in Large Language Models: A Survey.** `ACL findings` `2023`

  Jie Huang and Kevin Chen-Chuan Chang. [[Paper](https://aclanthology.org/2023.findings-acl.67)]

- **Let’s verify step by step.** `ICLR` `2023`

  Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. [[Paper](https://openreview.net/forum?id=v8L0pN6EOi)]

### 3 How to improve LLM-as-a-Judge?

#### 3.1 Design Strategy of Evaluation Prompts

##### Few-shot promping

- **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.** `EMNLP` `2023`

  Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. [[Paper](https://aclanthology.org/2023.emnlp-main.741)]

- **SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models.** `ACL findings` `2024`

  Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. [[Paper](https://aclanthology.org/2024.findings-acl.235)]

- **GPTScore: Evaluate as You Desire.** `NAACL` `2024`

  Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. [[Paper](https://aclanthology.org/2024.naacl-long.365)]

##### Evaluation steps decomposition

- **G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment.** `EMNLP` `2023`

  Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. [[Paper](https://aclanthology.org/2023.emnlp-main.153)]

- **DHP Benchmark: Are LLMs Good NLG Evaluators?** `ArXiv preprint` `2024`

  Yicheng Wang, Jiayi Yuan, Yu-Neng Chuang, Zhuoer Wang, Yingchi Liu, Mark Cusick, Param Kulkarni, Zhengping Ji, Yasser Ibrahim, and Xia Hu. [[Paper](https://arxiv.org/abs/2408.13704)]

- **SocREval: Large Language Models with the Socratic Method for Reference-free Reasoning Evaluation.** `NAACL findings` `2024`

  Hangfeng He, Hongming Zhang, and Dan Roth. [[Paper](https://aclanthology.org/2024.findings-naacl.175)]

- **Branch-Solve-Merge Improves Large Language Model Evaluation and Generation.** `NAACL` `2024`

  Swarnadeep Saha, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, and Xian Li. [[Paper](https://aclanthology.org/2024.naacl-long.462)]

##### Evaluation criteria decomposition

- **HD-Eval: Aligning Large Language Model Evaluators Through Hierarchical Criteria Decomposition.** `ACL` `2024`

  Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. [[Paper](https://aclanthology.org/2024.acl-long.413)]

- **Are LLM-based Evaluators Confusing NLG Quality Criteria?** `ACL` `2024`

  Xinyu Hu, Mingqi Gao, Sen Hu, Yang Zhang, Yicheng Chen, Teng Xu, and Xiaojun Wan. [[Paper](https://aclanthology.org/2024.acl-long.516)]

##### Shuffling contents

- **Large language models are not fair evaluators.** `ACL` `2024`

  Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. [[Paper](https://aclanthology.org/2024.acl-long.511)]

- **Generative judge for evaluating alignment.** `ArXiv preprint` `2023`

  Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. [[Paper](https://arxiv.org/abs/2310.05470)]

- **Judgelm: Fine-tuned large language models are scalable judges.** `ArXiv preprint` `2023`

  Lianghui Zhu, Xinggang Wang, and Xinlong Wang. [[Paper](https://arxiv.org/abs/2310.17631)]

- **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** `ArXiv preprint` `2023`

  Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. [[Paper](https://arxiv.org/abs/2306.05087)]

##### Conversion of evaluation tasks

- **Aligning with human judgement: The role of pairwise preference in large language model evaluators. ** `COLM` `2024`

  Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulic, Anna Korhonen, and Nigel Collier. [[Paper](https://arxiv.org/abs/2403.16950)]

##### Constraining outputs in structured formats

- **G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment.** `EMNLP` `2023`

  Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. [[Paper](https://aclanthology.org/2023.emnlp-main.153)]

- **DHP Benchmark: Are LLMs Good NLG Evaluators?** `ArXiv preprint` `2024`

  Yicheng Wang, Jiayi Yuan, Yu-Neng Chuang, Zhuoer Wang, Yingchi Liu, Mark Cusick, Param Kulkarni, Zhengping Ji, Yasser Ibrahim, and Xia Hu. [[Paper](https://arxiv.org/abs/2408.13704)]

- **LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models.** `NLP4ConvAI` `2023`

  Yen-Ting Lin and Yun-Nung Chen. [[Paper](https://aclanthology.org/2023.nlp4convai-1.5)]

##### Providing evaluations with explanations

- **CLAIR: Evaluating Image Captions with Large Language Models.** `EMNLP` `2023`

  David Chan, Suzanne Petryk, Joseph Gonzalez, Trevor Darrell, and John Canny. [[Paper](https://aclanthology.org/2023.emnlp-main.841)]

- **FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning Using a Large Multimodal Model.** `ACL` `2024`

  Yebin Lee, Imseong Park, and Myungjoo Kang. [[Paper](https://aclanthology.org/2024.acl-long.205)]

#### 3.2 Improvement Strategy of LLMs' Abilities

##### Fine-tuning via Meta Evaluation Dataset

- **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization.** `ArXiv preprint` `2023`

  Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. [[Paper](https://arxiv.org/abs/2306.05087)]

- **SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models.** `ACL findings` `2024`

  Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. [[Paper](https://aclanthology.org/2024.findings-acl.235)]

- **Offsetbias: Leveraging debiased data for tuning evaluators.** `ArXiv preprint` `2024`

  Junsoo Park, Seungyeon Jwa, Meiying Ren, Daeyoung Kim, and Sanghyuk Choi. [[Papaer](https://arxiv.org/abs/2407.06551)]

- **Judgelm: Fine-tuned large language models are scalable judges.** `ArXiv preprint` `2023`

  Lianghui Zhu, Xinggang Wang, and Xinlong Wang. [[Paper](https://arxiv.org/abs/2310.17631)]

- **CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation.** `ACL` `2024`

  Pei Ke, Bosi Wen, Andrew Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, et al. [[Paper](https://aclanthology.org/2024.acl-long.704)]

##### Iterative Optimization Based on Feedbacks

- **INSTRUCTSCORE: Towards Explainable Text Generation Evaluation with Automatic Feedback.** `EMNLP` `2023`

  Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag, William Wang, and Lei Li. [[Paper](https://aclanthology.org/2023.emnlp-main.365)]

- **Jade: A linguistics-based safety evaluation platform for llm.** `ArXiv preprint` `2023`

  Mi Zhang, Xudong Pan, and Min Yang. [[Paper](https://arxiv.org/abs/2311.00286)]

#### 3.3 Optimization Strategy of Final Results

##### Summarize by multiple rounds

- **Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks.** `EMNLP` `2023`

  Andrea Sottana, Bin Liang, Kai Zou, and Zheng Yuan. [[Paper](https://aclanthology.org/2023.emnlp-main.543)]

- **On the humanity of conversational ai: Evaluating the psychological portrayal of llms.** `ICLR` `2023`

  Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael Lyu. [[Paper](https://openreview.net/forum?id=H3UayAQWoE)]

- **Generative judge for evaluating alignment.** `ArXiv preprint` `2023`

  Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. [[Paper](https://arxiv.org/abs/2310.05470)]

##### Vote by multiple LLMs

- **Goal-Oriented Prompt Attack and Safety Evaluation for LLMs.** `ArXiv preprint` `2023`

  Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, and Fei Wu. [[Paper](https://arxiv.org/abs/2309.11830)]

- **Benchmarking Foundation Models with Language-Model-as-an-Examiner.** `NeurIPS` `2023`

  Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, and Lei Hou. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f64e55d03e2fe61aa4114e49cb654acb-Paper-Datasets_and_Benchmarks.pdf)]

##### Score smoothing

- **FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning Using a Large Multimodal Model.** `ACL` `2024`

  Yebin Lee, Imseong Park, and Myungjoo Kang. [[Paper](https://aclanthology.org/2024.acl-long.205)]

- **G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment.** `EMNLP` `2023`

  Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. [[Paper](https://aclanthology.org/2023.emnlp-main.153)]

- **DHP Benchmark: Are LLMs Good NLG Evaluators?** `ArXiv preprint` `2024`

  Yicheng Wang, Jiayi Yuan, Yu-Neng Chuang, Zhuoer Wang, Yingchi Liu, Mark Cusick, Param Kulkarni, Zhengping Ji, Yasser Ibrahim, and Xia Hu. [[Paper](https://arxiv.org/abs/2408.13704)]

##### Self validation

- **TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models.** `EMNLP` `2023`

  Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor. [[Paper](https://aclanthology.org/2023.emnlp-main.127)]

### 4 How to evaluate LLM-as-a-Judge？

#### 4.1 Basic Metric

- **Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.** `ArXiv preprint` `2024`

  Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. [[Paper](https://arxiv.org/abs/2406.12624)]

- **Benchmarking Foundation Models with Language-Model-as-an-Examiner.** `NeurIPS` `2023`

  Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, and Lei Hou. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f64e55d03e2fe61aa4114e49cb654acb-Paper-Datasets_and_Benchmarks.pdf)]

- **Aligning with human judgement: The role of pairwise preference in large language model evaluators. ** `COLM` `2024`

  Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulic, Anna Korhonen, and Nigel Collier. [[Paper](https://arxiv.org/abs/2403.16950)]

- *MTBench & Chatbot Arena Conversations*：**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** `NeurIPS` `2023`

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

- *FairEval*：**Large language models are not fair evaluators.** `ACL` `2024`

  Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. [[Paper](https://aclanthology.org/2024.acl-long.511)]

- *LLMBar*：**Evaluating Large Language Models at Evaluating Instruction Following.** `ArXiv preprint` `2023`

  Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. [[Paper](https://arxiv.org/abs/2310.07641)]

- **MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark.** `ICML` `2024`

  Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. [[Paper](https://openreview.net/forum?id=dbFEFHAD79)]

- **CodeJudge-Eval: Can Large Language Models be Good Judges in Code Understanding?** `COLING` `2025`

  Yuwei Zhao, Ziyang Luo, Yuchen Tian, Hongzhan Lin, Weixiang Yan, Annan Li, and Jing Ma. [[Paper](https://aclanthology.org/2025.coling-main.7)]

- *KUDGE*：**LLM-as-a-Judge & Reward Model: What They Can and Cannot Do.** `ArXiv preprint` `2024`

  Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, and Seunghyeok Hong. [[Paper](https://arxiv.org/abs/2409.11239)]

- *CALM*：**Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.** `ArXiv preprint` `2024`

  Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. [[Paper](https://arxiv.org/abs/2410.02736)]

- *LLMEval*$^2$：**Wider and deeper llm networks are fairer llm evaluators.** `ArXiv preprint` `2023`

  Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. [[Paper](https://arxiv.org/abs/2308.01862)]

#### 4.2 Bias

##### Position Bias

- **Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs.** `ArXiv preprint` `2024`

  Lin Shi, Weicheng Ma, and Soroush Vosoughi. [[Paper](https://arxiv.org/abs/2406.07791)]

- **Large language models are not fair evaluators.** `ACL` `2024`

  Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. [[Paper](https://aclanthology.org/2024.acl-long.511)]

- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.** `ArXiv preprint` `2024`

  Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. [[Paper](https://arxiv.org/abs/2410.02736)]

##### Length Bias

- **An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4** `ArXiv preprint` `2024`

  Hui Huang, Yingqi Qu, Xingyuan Bu, Hongli Zhou, Jing Liu, Muyun Yang, Bing Xu, Tiejun Zhao. [[Paper](https://arxiv.org/abs/2403.02839)]

- **Offsetbias: Leveraging debiased data for tuning evaluators.** `ArXiv preprint` `2024`

  Junsoo Park, Seungyeon Jwa, Meiying Ren, Daeyoung Kim, and Sanghyuk Choi. [[Papaer](https://arxiv.org/abs/2407.06551)]

- **Verbosity Bias in Preference Labeling by Large Language Models.** `ArXiv preprint` `2023`

  Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. [[Paper](https://arxiv.org/abs/2310.10076)]

##### Self-Enhancement Bias

- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.** `ArXiv preprint` `2024`

  Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. [[Paper](https://arxiv.org/abs/2410.02736)]

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** `NeurIPS` `2023`

  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)]

##### Other Bias

- **Humans or LLMs as the Judge? A Study on Judgement Bias.** `EMNLP` `2024`

  Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang. [[Paper](https://aclanthology.org/2024.emnlp-main.474)]

- **Subtle Biases Need Subtler Measures: Dual Metrics for Evaluating Representative and Affinity Bias in Large Language Models. ** `ACL` `2024`

  Abhishek Kumar, Sarfaroz Yunusov, Ali Emami. [[Paper](https://aclanthology.org/2024.acl-long.23/)]

- **Examining Query Sentiment Bias Effects on Search Results in Large Language Models.** `ESSIR` `2023`

  Alice Li, and Luanne Sinnamon. [[Paper](https://2023.essir.eu/FDIA/papers/FDIA_2023_paper_2.pdf)]

#### 4.3 Adversarial Robustness

- **Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment.** `EMNLP` `2024`

  Vyas Raina, Adian Liusie, Mark Gales. [[Paper](https://aclanthology.org/2024.emnlp-main.427)]

- **Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the effect of Epistemic Markers on LLM-based Evaluation.** `ArXiv preprint` `2024`

  Dongryeol Lee, Yerin Hwang, Yongil Kim, Joonsuk Park, and Kyomin Jung. [[Paper](https://arxiv.org/abs/2410.20774)]

- **Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates.** `ICLR` `2025`

  Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Jing Jiang, and Min Lin. [[Paper](https://arxiv.org/abs/2410.07137)] 

- **Benchmarking Cognitive Biases in Large Language Models as Evaluators.** `ACL Findings` `2024`

  Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. [[Paper](https://aclanthology.org/2024.findings-acl.29)]

- **Baseline Defenses for Adversarial Attacks Against Aligned Language Models.** `ArXiv preprint` `2023`

  Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. [[Paper](https://arxiv.org/abs/2309.00614)]
