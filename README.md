<h1>MGCIN</h1>
<h2>Model Structure</h2>
(1) Rumor information extraction module: Using the ConcasRTE model to extract subject relation object triplets from rumor events, providing basic information for subsequent common sense inference and graph convolution.
(2) Common sense reasoning module: Encode the extracted triplets and input them into the pre trained common sense reasoning module COMET to integrate common sense information and rumor information, generating feature vectors. This vector serves as both the input for the central network module and the basis for generating classification labels for the auxiliary module.
(3) Graph Convolutional Network Module: Through graph convolutional networks in both top-down and bottom-up directions, aggregation and updating of node features in rumor propagation graphs are achieved. The feature vectors in each direction integrate external common sense information, text information, and rumor propagation graph structure information.
(4) Classification module: Combine the losses of the auxiliary module and the main network module for joint training, and use model fusion methods to generate the final classification labels.

This is the data and code for our paper MGCIN: Multi-branch Graph Convolutional Inference Networks for Rumor Detection.

Prerequisites
Make sure your local environment has the following installed:

cuda version < 11.0
pytorch>=1.7.1 & <=1.9
wandb == 0.9.7
