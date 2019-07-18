# Support 996.icu
I am proudly in support of the 996.icu initiative which upholds the labor rights of Chinese IT practitioners that are being seriously violated. Check out more by clicking the link below.

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>

# IMPORTANT NOTES
We have discovered some code bugs in our source code, which have affected the experimental results reported in our paper to some extent. We have done our best to rectified the bugs and have updated our corrected code and experimental results in this repository. We are pleased to say that the vast majority of our findings in our paper still holds after rectification. We will discuss the expreimental results later in this readme file. We apologize for any inconvenience caused by our mistakes.

# Overview
This is the source code of PU-Shapelets, a pattern-based positive unlabeled classification algorithm for time series proposed in

	Shen Liang, Yanchun Zhang, Jiangang Ma: PU-Shapelets: Towards Pattern-Based Positive Unlabeled Classification of Time Series.
	DASFAA (1) 2019: 87-103


There are four directories in this repository:

PUSh: the C++ source code of our PU-Shapelets (PUSh) algorithm.

P1NN: the CUDA-C source code of the propagating one-nearest-neighbor (P1NN) baselines.

sample_data: sample datasets and a random seed generator for generating initial positive unlabeled examples.

results: the raw results (F-score and running time) on all datasets and the updated figures. We will elaborate on the experimental results later. Also, we have included the output files for each sample dataset, using the first of the pre-generated seeds. 

# Notes on our paper

Due to a lack of knowledge at the time of writing the paper, the following claims in our paper are questionable.

In the first paragraph of the introduction part, we claimed that "...To the best of our knowledge, no conventional
supervised TSC methods can be applied to such cases where only one class is labeled...". However, the following paper has recently come to our notice.

	Akihiro Yamaguchi, Takeichiro Nishikawa: One-Class Learning Time-Series Shapelets. BigData 2018: 2365-2372

The method presented in this paper learns shapelets using only the majority class. We have not yet had the time to read the paper more thoroughly, so we are not sure whether this method can train a classifier using only one class, but this certainly remains a possibility.

Also, in the third paragraph of the introduction part of our paper, we claimed that to the best of our knowledge, we were the first to take on PU discovery of shapelets. However, the following paper has recently come to our notice.

	Haishuai Wang, Qin Zhang, Jia Wu, Shirui Pan, Yixin Chen: Time Series Feature Learning with Labeled and Unlabeled Data.
	Pattern Recogonition 89: 55-66 (2019)
	
This paper deals with semi-supervised learning of shapelets. Again, we have not yet had the time to read the paper more thoroughly, so we are not sure whether it can handle PU data. Again, this remains a possibility. 

Last, we may have some confusions as to the precision-recall breakeven point (P-R breakeven point) metric. By definition, P-R breakeven point is the point where precision equals recall. We were able to deduce that following the U example ranking algorithms used in our paper as well as the baselines, the P-R breakeven point is the point where the number of labeled examples equals the actual number of positive examples in the training set, and it should be unique. However, it has come to our notice that in one of our baseline papers, namely

	Li Wei, Eamonn J. Keogh: Semi-supervised time series classification. KDD 2006: 748-753

the P-R breakeven point varies with the number of labeled examples. We currently have no idea how the authors of this paper were able to calculate the P-R breakeven point when the number of labeled examples is not equal to the actual number of positive examples. Maybe we have some misunderstandings as to what P-R breakeven point means in this paper.

We express our sincere apology for any confusion or inconvenience. We will more thoroughly study the aforementioned issues to reach a definitive conclusion. Please also advise us on these issues if you have any ideas on their solutions.


# Usage of source code

== On the datasets ==

We have provided two sample datasets. For more datasets, please download them from the UCR archive:

	Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen and Gustavo Batista (2015). 
	The UCR Time Series Classification Archive. URL www.cs.ucr.edu/~eamonn/time_series_data/

Note that we have conducted our experiments on the 2015 version of the UCR archive. For the latest version (2018), please see

	Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi , 
	Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu, Nurjahan Begum, Anthony Bagnall , 
	Abdullah Mueen and Gustavo Batista (2018). The UCR Time Series Classification Archive. 
	URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
	
There are certain differences between the two. Please refer to the UCR datasets briefing paper for details.

== On the implementation of the P1NN baselines

The P1NN baselines involve calculations of DTW distances. Brute-force calculation of DTW is painfully slow. Therefore, we have applied GPU acceleration. To run the P1NN baselines, you would need NVIDIA graphics card and CUDA toolkit. However, our PUSh algorithm is purely CPU-based. Also, for the P1NN online classification phase on the MALLAT dataset, it can still take a long time even with GPU acceleration. We strongly recomment beginning with the Car dataset.

== Generate initial positive labeled examples with generateSeeds.m ==

This is the random seed generator used to generate initial positive labeled examples.
We have attached the seeds used in our experiments in seeds_Car.txt and seeds_MALLAT.txt.
However, feel free to generate your own seeds with the seed generator. 
Nonetheless, note that the generated seeds will OVERWRITE the two files.

-- necessary parameters --

dataset: the dataset name

-- optional parameters --

numSeeds: the number of seeds to generate, default 10. Note that when numSeeds is set to more than the number of positive examples np, it is automatically reset to np.

path: the output path, default dataset (the dataset path)

===== Training PU-Shapelets with PUSh.cpp =====

-- necessary parameters --

datasetName: dataset name

numTrain: number of training examples

numP: number of positive training examples

numPLabeled: number of initial positive labeled examples

tsLen: time series length

seed_id: ID of the set of initial positive labeled examples

-- optional parameters --

maxNumSh: maximum number of (assumed) shapelets, default 200

minNumIters: minimum number of pattern ensemble iterations, default 5 if numP >= 10, 1 otherwise

maxNumIters: maximum number of pattern ensemble iterations, default numTrain * 2 / 3 - numPLabeled

minSLen: minimum shapelet length, default 10

maxSLen: maximum shapelet length, default tsLen

sLenStep: shapelet length step, default (maxSLen - minSLen) / 10

path: dataset path, default "..\\sample_data\\" + datasetName

outputPath: output path, default "..\\results"

===== Classification with Classify_PUSh.cpp =====

-- necessary parameters --

datasetName: dataset name

numTrain: number of training examples

numP: number of positive training examples

numPLabeled: number of initial positive labeled examples

tsLen: time series length

numTest: number of test examples

seed_id: ID of the set of initial positive labeled examples

-- optional parameters --

numSh: the number of shapelets used for classification, default 10

maxNumSh: maximum number of shapelets obtained from training, default 200. This parameter is set only to read the correct file.

minNumIters: minimum number of pattern ensemble iterations in training, default 5 if numP >= 10, 1 otherwise. This parameter is set only to read the correct file.

maxNumIters: maximum number of pattern ensemble iterations in training, default numTrain * 2 / 3 - numPLabeled. This parameter is set only to read the correct file.

maxSLen: maximum possible shapelet length, default tsLen

path: dataset path, default "..\\sample_data\\" + datasetName

outputPath: output path, default "..\\results"

===== Training P1NN with P1NN.cu =====

-- necessary parameters --

datasetName: dataset name

numTrain: number of training examples

numP: number of positive training examples

numPLabeled: number of initial positive labeled examples

tsLen: time series length

seed_id: ID of the set of initial positive labeled examples

-- optional parameters --

minNumIters: minimum number of P1NN iterations in training, default 5 if numP >= 10

maxNumIters: maximum number of P1NN iterations in training, default numTrain * 2 / 3 - numPLabeled

warp: the warping window for DTW and DTW-D, default 0, in which case we use the provided warping window on the UCR webpage. We will discuss this later.

maxThreadsPerBlock: the maximum number of threads per GPU block for DTW calculation, default 8. This setting is reserved for large datasets. For small datasets, it is advisable to increase this number for maximum performance. Please refer to the CUDA documentation for more.

maxBlocksPerGrid: the maximum number of block per GPU grid for DTW calculation, default 8. This setting is reserved for large datasets. For small datasets, it is advisable to increase this number for maximum performance. Please refer to the CUDA documentation for more.

dataInfoPath: the path to the file "InfoAll" which contains information on the 2015 version of the UCR datasets, default "..\\sample_data\\"

path: dataset path, default "..\\sample_data\\" + datasetName

outputPath: output path, default "..\\results"

===== Classification with Classify_P1NN.cu =====

-- necessary parameters --

datasetName: dataset name

numTrain: number of training examples

numP: number of positive training examples

numPLabeled: number of initial positive labeled examples

tsLen: time series length

numTest: number of test examples

seed_id: ID of the set of initial positive labeled examples

-- optional parameters --

minNumIters: minimum number of P1NN iterations in training, default 5 if numP >= 10. This parameter is set only to read the correct file.

maxNumIters: maximum number of P1NN iterations in training, default numTrain * 2 / 3 - numPLabeled. This parameter is set only to read the correct file.

maxThreadsPerBlock: the maximum number of threads per GPU block for DTW calculation, default 8. This setting is reserved for large datasets. For small datasets, it is advisable to increase this number for maximum performance. Please refer to the CUDA documentation for more.

maxBlocksPerGrid: the maximum number of block per GPU grid for DTW calculation, default 8. This setting is reserved for large datasets. For small datasets, it is advisable to increase this number for maximum performance. Please refer to the CUDA documentation for more.

dataInfoPath: the path to the file "InfoAll" which contains information on the 2015 version of the UCR datasets, default "..\\sample_data\\"

path: dataset path, default "..\\sample_data\\" + datasetName

outputPath: output path, default "..\\results"

# On the corrected experimental results

Due to some bugs in the original code (which has been corrected and updated in this repository), there are certain inaccuracies in the experimental results reported in our paper. The rectified results are stored in the "results" directory in this repository. Specifically, F-scores.xlsx reports the raw average F1-scores. Running time.xlsx reports the average running time of our PUSh algorithm. The .png files are updated figures which correspond to the ones appearing in the original paper. We will elaborate on them later.

===== On precision, recall and F1-score calculation =====

The calculation of precision (P), recall (R) and F1-score (F) can run into divide-by-zero situations, in which cases their values are ill-defined. In such cases, our implementation will set their values to -1. We measure the performances of all methods by their F1-scores in the training and classification phases. When F is -1, we reset it to 0 before calculating the average scores (reported in F-scores.xlsx) for comparison. Such cases occur when the true positive (TP) value is 0, meaning that either an algorithm fails to detect any positive examples in the unlabeled data, or none of the ones classified as positive are actually positive. Note that it is not always advisable to replace the -1 values to 0. For example, if true positive, false negative and false positive values are all 0, we should set P, R, F all to 1, since the algorithm correctly decides that there are no positive examples in the unlabeled data. However, this is impossible in our case, for there are always unlabeled positive examples, both in the U set of the training set and the testing set.

For the five GBTRM stopping criteria for the P1NN algorithm, sometimes they cannot determine a stopping point. In such cases, our implementation will set both the training and classification F values to -2, indicating that the stopping criterion is ineffective on this dataset. Such F values are reset to 0 before calculating the average scores (reported in F-scores.xlsx) for comparison.

===== On ploting the critical difference diagram =====

To plot the critical difference diagrams, we use 1-F as the input metric, which is equal to Van Rijsbergen's effectiveness measure with alpha = 0.5. See https://en.wikipedia.org/wiki/F1_score for more.

We display the diagram for our PUSh and top-10 ranking baseline methods. By top-10 baseline methods, we mean that when all 25 methods are displayed in one diagram, these methods are the first 10 baseline methods (excluding PUSh) to the right of the diagram.

===== On the P1NN results =====

Recall that our PUSh algorithm requires setting of th lower and upper bounds of the number of positive and unlabeled examples. For fairness to the P1NN algorithms, we independently conduct two experiments on them. In one of them, we set the same bounds as PUSh. In the other, we set no bounds, which means the minimum allowed number of positive and unlabeled examples is 1, and the maximum number is numTrain - 1 - numPLabeled where numTrain is the training set size and numPLabeled is the number of initially labeled examples. We select the better one from these two runs for final comparison. 

In each one of these two runs, for certain datasets, there are a range of available DTW warping windows to choose from for DTW and DTW-D. Also, for the WK criterion, we can pick from a range of possible stopping points. In such cases, we pick the setting (DTW warping window, stopping point or the combination of both) that yield the best training performance, and report the training and testing performances under such a setting. Please refer to our source code for details.

===== On the performance of U example ranking methods ====

The corrected figures reporting U example ranking methods (Fig. 8 in the original paper) are provided as raw_trans_oracle.png (raw results) and cd_trans_oracle.png (critical difference diagram). Our conclusion in the paper still holds: There are no significant differences among the performances of the three baseline methods. Our PE outperforms them all and significantly outperforms DTW.

===== On the performance of labeling the U set =====

The corrected figures reporting U set labeling methods (Fig. 9 in the original paper) are provided as raw_trans_sc.png (raw results) and cd_trans_sc.png (critical difference diagram). Our conclusion in the paper still holds: No significant difference is observed among the baselines. PUSh significantly outperforms the top-10 baselines. Most top ranking baselines utilize one of G1-G5. However, contrary to our analysis in our original paper, we now suspect that maybe the performance of G1-G5 is not so impressive after all. The reason is that out of 24 baselines, 15 apply one of G1-G5. The large number of baselines utilizing G1-G5 means that it is no surprise many baselines with G1-G5 end up in the top-10. By contrast, out of the three baselines using R, two makes it into the top-10, and all three baselines with W are among the top-10. Note however that while it appears that the baselines with W performs well among all baselines. recall that we have favored W in the experimental settings, and the advantage displayed here may not still hold when such bias is withdrawn.

===== On the performance of Online Classification =====

The corrected figures reporting U set labeling methods (Fig. 10 in the original paper) are provided as raw_induc_sc.png (raw results) and cd_induc_sc.png (critical difference diagram). Our conclusions in the paper still holds. First, the number of shapelets does not significantly affect the classification performance of PE. We have set this value to 10 : 10 : 50 and performed pairwise Wilcoxon signed rank test. The minimum p-value is 0.0642 which is greater than the 0.05 threshold. We still set this value to 10 in the following experiments. Second, most of the top ranking methods in the U example labeling process remain highly competitive. No significant difference is observed among the top-10 baselines and our PUSh significantly outperforms 9 out of 10 of them except DTWD+R.

===== On the running time =====

The corrected figures reporting running time (Fig. 11 in the original paper) are provided as "train time.png" and "test time.png". For the training time, our conclusion in the paper still holds: PUSh is able to achieve reasonable running time, with the longest average running time a little over 1000 seconds. For the testing time, our original paper got the x-axis ticks as well as the time on the MALLAT dataset wrong. Compared to the incorrect version, the corrested figure actually shows a more promissing result with all datasets taking less than 0.01s to classify a future example. Also, the MALLAT dataset is somewhat an outlier, whose running time does not follow the quasi-linear pattern displayed across other datasets. The possible explanations are two-fold: On the one hand, the shapelets used on the MALLAT datasets are all of the shortest possible length (which is 10), which is far shorter than the full time series length (1024). Second, these shapelets are selected from positions close to the beginning of some training examples, and are likely to match early positions in the testing examples. In our implementation, we have adopted an early abondoning strategy used in Ye and Keogh's original shapelet paper to accelerate calculation, and this strategy is likely to yield more gain in such cases where the matching point is among the early positions.
