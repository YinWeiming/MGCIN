<h1>MGCIN</h1>
<h2>Model Structure</h2>
(1) Rumor information extraction module: Using the ConcasRTE model to extract subject relation object triplets from rumor events, providing basic information for subsequent common sense inference and graph convolution.<br>
(2) Common sense reasoning module: Encode the extracted triplets and input them into the pre trained common sense reasoning module COMET to integrate common sense information and rumor information, generating feature vectors. This vector serves as both the input for the central network module and the basis for generating classification labels for the auxiliary module.<br>
(3) Graph Convolutional Network Module: Through graph convolutional networks in both top-down and bottom-up directions, aggregation and updating of node features in rumor propagation graphs are achieved. The feature vectors in each direction integrate external common sense information, text information, and rumor propagation graph structure information.<br>
(4) Classification module: Combine the losses of the auxiliary module and the main network module for joint training, and use model fusion methods to generate the final classification labels.<br>

<b>This is the data and code for our paper MGCIN: Multi-branch Graph Convolutional Inference Networks for Rumor Detection.</b>

<h2>Prerequisites</h2>
Make sure your local environment has the following installed:<br>
cuda version < 11.0 <br>
pytorch>=1.7.1 & <=1.9<br>
wandb == 0.9.7<br>
python==3.5.2<br>
numpy== 1.18.1<br>
torch_scatter==1.4.0<br>
torch_sparse==0.4.3<br>
torch_cluster==1.4.5<br>
torch_geometric==1.3.2<br>
tqdm==4.40.0<br>
joblib==0.14.1<br>

<h2>Datastes</h2>
<strong>We provide the dataset in the data folder:</strong><br>
The datasets used in the experiments were based on the three publicly available datasets Twitter15,Twitter16,PHEME.<br>

Wu K,Yang S,Zhu K Q.False Rumors Detection on Sina Weibo by Propagation Structures[C]//Proceedings of 2015 IEEE 31st International Conference on Data Engineering. 2015: 651-662.<br>
Ma J, Gao W, Wong K F. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning[C]//Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017:708-717.<br>
Zubiaga A, Liakata M, Procter R, et al. Analysing how people orient to and spread rumours in    social media by looking at conversational threads[J]. PloS one, 2016, 11(3): e0150989.<br>



