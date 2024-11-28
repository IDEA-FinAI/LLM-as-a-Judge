# Awesome LLM-as-a-Judge

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



### Overview of LLM-as-a-Judge

![overview](F:/Work/IDEA/工作内容/LLM as Evaluator/LLM-as-a-Judge/images/overview.jpg)



### Evaluation Pipelines of LLM-as-a-Judge

![evaluation_pipeline](F:/Work/IDEA/工作内容/LLM as Evaluator/LLM-as-a-Judge/images/evaluation_pipeline.jpg)



### Improvement Strategies for LLM-as-a-Judge

![improvement_strategy](F:/Work/IDEA/工作内容/LLM as Evaluator/LLM-as-a-Judge/images/improvement_strategy.jpg)



### Paper List

#### 1.Survey

1. **Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges** `Preprint`

   *Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, Dieuwke Hupkes* [[Paper](https://arxiv.org/abs/2406.12624)] [[Code](https://github.com/UMass-Meta-LLM-Eval/llm_eval)], 2024.07

#### 2.Analysis

1. **A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators** `AAAI 2024`

   *Chen Zhang, Luis Fernando D'Haro, Yiming Chen, Malu Zhang, Haizhou Li* [[Paper](https://arxiv.org/abs/2312.15407)] [[Code](https://github.com/e0397123/comp-analysis)], 2024.01

2. **Large Language Models Cannot Self-Correct Reasoning Yet** `ICLR 2024`

   *Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou* [[Paper](https://arxiv.org/abs/2310.01798)], 2024.05

3. **Large Language Models are not Fair Evaluators** `ACL 2024`

   *Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, Zhifang Sui* [[Paper](https://aclanthology.org/2024.acl-long.511)] [[Code](https://github.com/i-Eval/FairEval)], 2023.08

4. **Subtle Biases Need Subtler Measures: Dual Metrics for Evaluating Representative and Affinity Bias in Large Language Models** `ACL 2024`

   *Abhishek Kumar, Sarfaroz Yunusov, Ali Emami* [[Paper](https://arxiv.org/abs/2405.14555)] [[Code](https://github.com/akkeshav/subtleBias)], 2024.06

5. **Are LLM-based Evaluators Confusing NLG Quality Criteria** `ACL 2024`

   *Xinyu Hu, Mingqi Gao, Sen Hu, Yang Zhang, Yicheng Chen, Teng Xu, Xiaojun Wan* [[Paper](https://aclanthology.org/2024.acl-long.516)] [[Code](https://github.com/herrxy/LLM-evaluator-reliability)], 2024.06

6. **Likelihood-based Mitigation of Evaluation Bias in Large Language Models** `ACL 2024 findings`

   *Masanari Ohi, Masahiro Kaneko, Ryuto Koike, Mengsay Loem, Naoaki Okazaki* [[Paper](https://aclanthology.org/2024.findings-acl.193)], 2024.05

7. **Can Large Language Models Be an Alternative to Human Evaluations?** `ACL 2023`

   *Cheng-Han Chiang, Hung-yi Lee* [[Paper](https://aclanthology.org/2023.acl-long.870)], 2023.05

8. **Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks** `EMNLP 2023`

   *Andrea Sottana, Bin Liang, Kai Zou, Zheng Yuan* [[Paper](https://aclanthology.org/2023.emnlp-main.543)] [[Code](https://github.com/protagolabs/seq2seq_llm_evaluation)], 2023.10

9. **Is ChatGPT a Good NLG Evaluator? A Preliminary Study** `NewSumm @ EMNLP 2023`

   *Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, Jie Zhou* [[Paper](https://arxiv.org/abs/2303.04048)] [[Code](https://github.com/krystalan/chatgpt_as_nlg_evaluator)], 2023.10

10. **Comparing Two Model Designs for Clinical Note Generation; Is an LLM a Useful Evaluator of Consistency?**  `NAACL 2024 findings`

    *Nathan Brake, Thomas Schaaf* [[Paper](https://aclanthology.org/2024.findings-naacl.25)], 2024.04

11. **Is LLM a Reliable Reviewer? A Comprehensive Evaluation of LLM on Automatic Paper Reviewing Tasks** `COLING 2024`

    *Ruiyang Zhou, Lu Chen, Kai Yu* [[Paper](https://aclanthology.org/2024.lrec-main.816)] [[Dataset](https://huggingface.co/datasets/zhouruiyang/RR-MCQ)], 2024.05

12. **Exploring the Use of Large Language Models for Reference-Free Text Quality Evaluation: An Empirical Study** `Preprint`

    *Yi Chen, Rui Wang, Haiyun Jiang, Shuming Shi, Ruifeng Xu* [[Paper](https://arxiv.org/abs/2304.00723)] [[Code](https://github.com/MilkWhite/LLMs_for_Reference_Free_Text_Quality_Evaluation)], 2023.09

13. **Humans or LLMs as the Judge? A Study on Judgement Biases** `Preprint`

    *Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang* [[Paper](https://arxiv.org/abs/2402.10669)], 2024.06

14. **On the Limitations of Fine-tuned Judge Models for LLM Evaluation** `Preprint`

    *Hui Huang, Yingqi Qu, Hongli Zhou, Jing Liu, Muyun Yang, Bing Xu, Tiejun Zhao* [[Paper](https://arxiv.org/abs/2403.02839)] [[Code](https://github.com/HuihuiChyan/UnlimitedJudge)], 2024.06

15. **Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment** `Preprint`

    *Vyas Raina, Adian Liusie, Mark Gales* [[Paper](https://arxiv.org/abs/2402.14016)] [[Code](https://github.com/rainavyas/attack-comparative-assessment)], 2024.07

16. **Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates** `Preprint`

    *Hui Wei, Shenghua He, Tian Xia, Andy Wong, Jingyang Lin, Mei Han* [[Paper](https://arxiv.org/abs/2408.13006)] [[Code](https://github.com/shenghh2015/llm-judge-eval)], 2024.08

#### 3.Auto-Evaluator

1. **On the Humanity of Conversational AI: Evaluating the Psychological Portrayal of LLMs** `ICLR 2024 (oral)`

   *Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu* [[Paper](https://openreview.net/pdf?id=H3UayAQWoE)] [[Code](https://github.com/CUHK-ARISE/PsychoBench)], 2023.12

2. **Generative Judge for Evaluating Alignment** `ICLR 2024`

   *Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, Pengfei Liu* [[Paper](https://arxiv.org/abs/2310.05470)] [[Code](https://github.com/GAIR-NLP/auto-j)], 2023.12

3. **PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization** `ICLR 2024`

   *Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, Yue Zhang* [[Paper](https://arxiv.org/abs/2306.05087)] [[Code](https://github.com/WeOpenML/PandaLM)], 2024.05

4. **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph** `ICLR 2024`

   *Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, Jian Guo* [[Paper](https://arxiv.org/abs/2307.07697)] [[Code](https://github.com/IDEA-FinAI/ToG)], 2024.05

5. **HD-Eval: Aligning Large Language Model Evaluators Through Hierarchical Criteria Decomposition** `ACL 2024`

   *Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang* [[Paper](https://aclanthology.org/2024.acl-long.413)], 2024.02

6. **Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation** `ACL 2024`

   *Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Lifeng Jin, Linfeng Song, Haitao Mi, Helen Meng* [[Paper](https://aclanthology.org/2024.acl-long.107)] [[Code](https://github.com/zhangxy-2019/Self-Alignment-for-Factuality)], 2024.06

7. **FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning Using a Large Multimodal Model** `ACL 2024`

   *Yebin Lee, Imseong Park, Myungjoo Kang* [[Paper](https://aclanthology.org/2024.acl-long.205)] [[Code](https://github.com/Yebin46/FLEUR)], 2024.06

8. **KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models ** `ACL 2024`

   *Zhuohao Yu, Chang Gao, Wenjin Yao, Yidong Wang, Wei Ye, Jindong Wang, Xing Xie, Yue Zhang, Shikun Zhang* [[Paper](https://aclanthology.org/2024.acl-long.325/)] [[Code](https://github.com/zhuohaoyu/KIEval)], 2024.06

9. **ProxyQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models** `ACL 2024`

   *Haochen Tan, Zhijiang Guo, Zhan Shi, Lu Xu, Zhili Liu, Yunlong Feng, Xiaoguang Li, Yasheng Wang, Lifeng Shang, Qun Liu, Linqi Song* [[Paper](https://aclanthology.org/2024.acl-long.368)] [[Code](https://github.com/Namco0816/ProxyQA)], 2024.06

10. **CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation** `ACL 2024`

    *Pei Ke, Bosi Wen, Andrew Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang* [[Paper](https://aclanthology.org/2024.acl-long.704)] [[Code](https://github.com/thu-coai/CritiqueLLM)], 2024.06

11. **Aligning Large Language Models by On-Policy Self-Judgment** `ACL 2024`

    *Sangkyu Lee, Sungdong Kim, Ashkan Yousefpour, Minjoon Seo, Kang Min Yoo, Youngjae Yu* [[Paper](https://aclanthology.org/2024.acl-long.617)] [[Code](https://github.com/oddqueue/self-judge)], 2024.06

12. **FineSurE: Fine-grained Summarization Evaluation using LLMs** `ACL 2024`

    *Hwanjun Song, Hang Su, Igor Shalyminov, Jason Cai, Saab Mansour* [[Paper](https://aclanthology.org/2024.acl-long.51/)] [[Code](https://github.com/DISL-Lab/FineSurE-ACL24)], 2024.07

13. **SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models** `ACL 2024 findings`

    *Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao* [[Paper](https://arxiv.org/abs/2402.05044)] [[Code](https://github.com/OpenSafetyLab/SALAD-BENCH)], 2024.06

14. **LLM-EVAL: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models** `NLP4ConvAI @ ACL 2023`

    *Yen-Ting Lin, Yun-Nung Chen* [[Paper](https://aclanthology.org/2023.nlp4convai-1.5)], 2023.05

15. **G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment** `EMNLP 2023`

    *Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu* [[Paper](https://aclanthology.org/2023.emnlp-main.153)] [[Code](https://github.com/nlpyang/geval)], 2023.05

16. **TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models** `EMNLP 2023`

    *Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, Idan Szpektor* [[Paper](https://aclanthology.org/2023.emnlp-main.127)] [[Code](https://github.com/google-research/google-research/tree/master/true_teacher)], 2023.10

17. **INSTRUCTSCORE: Towards Explainable Text Generation Evaluation with Automatic Feedback **`EMNLP 2023`

    *Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag, William Wang, Lei Li* [[Paper](https://aclanthology.org/2023.emnlp-main.365)], 2023.10

18. **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation ** `EMNLP 2023`

    *Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi* [[Paper](https://aclanthology.org/2023.emnlp-main.741)] [[Code](https://github.com/shmsw25/FActScore)], 2023.10

19. **Revisiting Automated Topic Model Evaluation with Large Language Models** `EMNLP 2023 (short)`

    *Dominik Stammbach, Vilém Zouhar, Alexander Hoyle, Mrinmaya Sachan, Elliott Ash* [[Paper](https://aclanthology.org/2023.emnlp-main.581)] [[Code](https://github.com/dominiksinsaarland/evaluating-topic-model-output)], 2023.10

20. **CLAIR: Evaluating Image Captions with Large Language Models** `EMNLP 2023 (short)`

    *David Chan, Suzanne Petryk, Joseph Gonzalez, Trevor Darrell, John Canny* [[Paper](https://aclanthology.org/2023.emnlp-main.841)] [[Code](https://github.com/davidmchan/clair)], 2023.10

21. **GENRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models** `NAACL 2024`

    *Pengcheng Jiang, Jiacheng Lin, Zifeng Wang, Jimeng Sun, Jiawei Han* [[Paper](https://aclanthology.org/2024.naacl-long.155)] [[Code](https://github.com/pat-jj/GenRES)], 2024.02

22. **GPTScore: Evaluate as You Desire**  `NAACL 2024`

    *Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, Pengfei Liu* [[Paper](https://aclanthology.org/2024.naacl-long.365)] [[Code](https://github.com/jinlanfu/GPTScore)], 2023.02

23. **Branch-Solve-Merge Improves Large Language Model Evaluation and Generation**  `NAACL 2024`

    *Swarnadeep Saha, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, Xian Li* [[Paper](https://aclanthology.org/2024.naacl-long.462)], 2024.06

24. **A Multi-Aspect Framework for Counter Narrative Evaluation using Large Language Models**  `NAACL 2024 (short)`

    *Jaylen Jones, Lingbo Mo, Eric Fosler-Lussier, Huan Sun* [[Paper](https://aclanthology.org/2024.naacl-short.14)] [[Code](https://github.com/OSU-NLP-Group/LLM-CN-Eval)], 2024.03

25. **SocREval: Large Language Models with the Socratic Method for Reference-free Reasoning Evaluation**  `NAACL 2024 findings`

    *Hangfeng He, Hongming Zhang, Dan Roth* [[Paper](https://aclanthology.org/2024.findings-naacl.175)] [[Code](https://github.com/HornHehhf/SocREval)], 2024.06

26. **Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators** `COLM 2024`

    *Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulić, Anna Korhonen, Nigel Collier* [[Paper](https://arxiv.org/abs/2403.16950)] [[Code](https://github.com/cambridgeltl/PairS)], 2024.08

27. **LLMScore: Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation** `NeurIPS 2023`

    *Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, William Yang Wang* [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/47f30d67bce3e9824928267e9355420f-Abstract-Conference.html)] [[Code](https://github.com/YujieLu10/LLMScore)], 2023.05

28. **Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning** `NeurIPS 2023`

    *Beichen Zhang, Kun Zhou, Xilin Wei, Xin Zhao, Jing Sha, Shijin Wang, Ji-Rong Wen* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4a47dd69242d5af908cdd5d51c971cbf-Abstract-Datasets_and_Benchmarks.html)] [[Code](https://github.com/RUCAIBox/CARP)], 2023.06

29. **RRHF: Rank Responses to Align Language Models with Human Feedback without tears** `NeurIPS 2023`

    *Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang* [[Paper](https://arxiv.org/abs/2304.05302)] [[Code](https://github.com/GanjinZero/RRHF)], 2023.10

30. **Reflexion: Language Agents with Verbal Reinforcement Learning** `NeuralIPS 2023`

    *Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao* [[Paper](https://arxiv.org/abs/2303.11366)] [[Code](https://github.com/noahshinn024/reflexion)], 2023.10

31. **Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation** `NeurIPS 2023`

    *Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)] [[Code](https://github.com/evalplus/evalplus)], 2023.10

32. **Self-Evaluation Guided Beam Search for Reasoning** `NeurIPS 2023`

    *Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, Michael Xie* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html)] [[Code](https://github.com/YuxiXie/SelfEval-Guided-Decoding)], 2023.10

33. **Benchmarking Foundation Models with Language-Model-as-an-Examiner** `NeurIPS 2023`

    *Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, Lei Hou* [[Paper](https://arxiv.org/abs/2306.04181)] [[Code](http://lmexam.xlore.cn/)], 2023.11

34. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** `NeurIPS 2023`

    *Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica* [[Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)] [[Code](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)], 2023.12

35. **Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality** `Blog`

    *The Vicuna Team* [[Code](https://github.com/lm-sys/FastChat)] [[Blog](https://lmsys.org/blog/2023-03-30-vicuna/)], 2023.03

36. **Human-like summarization evaluation with chatgpt** `Preprint`

    *Mingqi Gao, Jie Ruan, Renliang Sun, Xunjian Yin, Shiping Yang, Xiaojun Wan* [[Paper](https://arxiv.org/abs/2304.02554)], 2023.04

37. **WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct** `Preprint`

    *Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Dongmei Zhang* [[Paper](https://arxiv.org/abs/2308.09583)] [[Code](https://github.com/nlpxucan/WizardLM)] [[Model](https://huggingface.co/WizardLM)], 2023.08

38. **Judgelm: Fine-tuned large language models are scalable judges ** `Preprint`

    *Lianghui Zhu, Xinggang Wang, Xinlong Wang* [[Paper](https://arxiv.org/abs/2310.17631)] [[Code](https://github.com/baaivision/JudgeLM)], 2023.10

39. **Goal-Oriented Prompt Attack and Safety Evaluation for LLMs** `Preprint`

    *Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu* [[Paper](https://arxiv.org/abs/2309.11830)] [[Code](https://github.com/liuchengyuan123/CPAD)], 2023.12

40. **JADE: A Linguistics-based Safety Evaluation Platform for Large Language Models** `Preprint`

    *Mi Zhang, Xudong Pan, Min Yang* [[Paper](https://arxiv.org/abs/2311.00286)] [[Code](https://github.com/whitzard-ai/jade-db)], 2023.12

41. **Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators** `Preprint`

    *Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto* [[Paper](https://arxiv.org/abs/2404.04475)] [[Code](https://github.com/tatsu-lab/alpaca_eval)], 2024.04

42. **OffsetBias: Leveraging Debiased Data for Tuning Evaluators** `Preprint`

    *Junsoo Park, Seungyeon Jwa, Meiying Ren, Daeyoung Kim, Sanghyuk Choi* [[Paper](https://www.arxiv.org/abs/2407.06551)] [[Code](https://github.com/ncsoft/offsetbias)], 2024.07

43. **DHP Benchmark: Are LLMs Good NLG Evaluators?** `Preprint`

    *Yicheng Wang, Jiayi Yuan, Yu-Neng Chuang, Zhuoer Wang, Yingchi Liu, Mark Cusick, Param Kulkarni, Zhengping Ji, Yasser Ibrahim, Xia Hu* [[Paper](https://arxiv.org/abs/2408.13704)], 2024.08

44. **Generative Verifiers: Reward Modeling as Next-Token Prediction** `Preprint`

    *Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal* [[Paper](https://arxiv.org/abs/2408.15240)], 2024.08

45. **Towards Leveraging Large Language Models for Automated Medical Q&A Evaluation** `Preprint`

    *Jack Krolik, Herprit Mahal, Feroz Ahmad, Gaurav Trivedi, Bahador Saket* [[Paper](https://arxiv.org/abs/2409.01941)], 2024.09

46. **LLMs as Evaluators: A Novel Approach to Evaluate Bug Report Summarization** `Preprint`

    *Abhishek Kumar, Sonia Haiduc, Partha Pratim Das, Partha Pratim Chakrabart*i [[Paper](https://arxiv.org/abs/2409.00630)] [[Code](https://bit.ly/3zk7qZr)], 2024.09

## Reasoning

1. **Reasoning with Language Model is Planning with World Model** `EMNLP 2023`

   *Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, Zhiting Hu* [[Paper](https://arxiv.org/abs/2305.14992)] [[Code](https://github.com/Ber666/RAP)] [[Reasoners](https://github.com/Ber666/llm-reasoners)] [[Blog](https://geyuyao.com/post/rap/)], 2023.05

2. **Solving Math Word Problems via Cooperative Reasoning induced Language Models** `ACL 2023`

   *Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, Yujiu Yang* [[Paper](https://aclanthology.org/2023.acl-long.245)] [[Code](https://github.com/TianHongZXY/CoRe)], 2023.07

3. **Human-like Few-Shot Learning via Bayesian Reasoning over Natural Language** `NeurIPS 2023`

   *Kevin Ellis* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2aa9b18b9ab37b0ab1fdaae46fb781d4-Abstract-Conference.html)] [[Code](https://github.com/ellisk42/humanlike_fewshot_learning)], 2023.09

4. **Deductive Verification of Chain-of-Thought Reasoning** `NeurIPS 2023`

   *Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, Hao Su* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html)] [[Code](https://github.com/lz1oceani/verify_cot)], 2023.10

5. **Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning** `NeurIPS 2023`

   *Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)] [[Code](https://github.com/iLampard/lamp)], 2023.10

6. **DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models** `NeurIPS 2023`

   *Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, Sibei Yang* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html)] [[Code](https://github.com/SooLab/DDCOT)], 2023.10

7. **Learning to Reason and Memorize with Self-Notes** `NeurIPS 2023`

   *Jack Lanchantin, Shubham Toshniwal, Jason Weston, arthur szlam, Sainbayar Sukhbaatar* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/274d0146144643ee2459a602123c60ff-Abstract-Conference.html)], 2023.10

8. **Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning** `NeurIPS 2023`

   *Xiaoqian Wu, Yong-Lu Li, Jianhua Sun, Cewu Lu* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)] [[Code](https://github.com/enlighten0707/Symbol-LLM)], 2023.11

9. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** `NeurIPS 2023`

   *Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, Karthik Narasimhan* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)], 2023.12

10. **Understanding Social Reasoning in Language Models with Language Models** `NeurIPS 2023`

    *Kanishk Gandhi, Jan-Philipp Fraenken, Tobias Gerstenberg, Noah Goodman* [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2b9efb085d3829a2aadffab63ba206de-Abstract-Datasets_and_Benchmarks.html)] [[Code](https://github.com/cicl-stanford/procedural-evals-tom)], 2023.12

11. **Automatic model selection with large language models for reasoning** `EMNLP 2023 findings`

    *James Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, Michael Xie* [[Paper](https://aclanthology.org/2023.findings-emnlp.55/)] [[Code](https://github.com/XuZhao0/Model-Selection-Reasoning)], 2023.10

12. **Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation** `Preprint`

    *Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, Dongmei Zhang* [[Paper](https://arxiv.org/abs/2311.04254)] [[Code](https://github.com/microsoft/Everything-of-Thoughts-XoT)], 2024.02

13. **Math-Shepherd: A Label-Free Step-by-Step Verifier for LLMs in Mathematical Reasoning** `Preprint`

    *Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, Zhifang Sui* [[Paper](https://arxiv.org/abs/2312.08935)], 2024.02

14. **Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents** `Preprint`

    *Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, Rafael Rafailov* [[Paper](https://arxiv.org/abs/2408.07199)], 2024.08
