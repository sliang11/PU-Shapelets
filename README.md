# Support 996.icu
I am proudly in support of the 996.icu initiative which upholds the labor rights of Chinese IT practitioners that are being seriously violated. Check out more by clicking the link below.

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>

# IMPORTANT NOTES
We have recently discovered some code bugs in our source code, which may very well have lead to incorrect experiment results. We are currently correcting these mistakes. We will update the code as well as the experimental results in this repository as soon as possible. We sincerely apologize for the inconvenience.

# PU-Shapelets
This is the source code of PU-Shapelets, a pattern-based positive unlabeled classification algorithm for time series proposed in

	Shen Liang, Yanchun Zhang, Jiangang Ma: PU-Shapelets: Towards Pattern-Based Positive Unlabeled Classification of Time Series.
	DASFAA (1) 2019: 87-103


There are two directories in this repository:

src: the source code of our PU-Shapelets.

sample_data: sample datasets and a random seed generator for generating initial positive unlabeled examples. The results will also be stored in the dataset directories by default. We have included sample test results for each dataset, using the first of the pre-generated seeds.

In addition, all our raw experimental results can be found in Fscores.xlsx and Running time.xlsx.

***** Notes on our paper *****

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


***** How to use the source code *****

== Note to users ==

We have provided two sample datasets. For more datasets, please download them from the UCR archive:

	Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen and Gustavo Batista (2015). 
	The UCR Time Series Classification Archive. URL www.cs.ucr.edu/~eamonn/time_series_data/

Note that we have conducted our experiments on the 2015 version of the UCR archive. For the latest version (2018), please see

	Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi , 
	Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu, Nurjahan Begum, Anthony Bagnall , 
	Abdullah Mueen and Gustavo Batista (2018). The UCR Time Series Classification Archive. 
	URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
	
There are certain differences between the two.

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

path: input and output path, default "..\\sample_data\\" + datasetName

===== Classification with classify_PUSh.cpp =====

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

path: input and output path, default "..\\sample_data\\" + datasetName
