# We provide a demo of TIB method based on CMU-mosi dataset.  Run main.py to show the training process of the model.

requirement:
pytorch==1.0.0
numpy==1.19.1
tqdm==4.48.0
scipy==1.1.0

   

## TIB:Cross-modal Emotion Clustering via Twin Information Bottleneck Method
> **Authors:**
Anonymous
<!-- > -->
## 1. Framework


The architecture of the proposed TIB is shown in Figure 2.

As shown in this figure, TIB consists of the following main components: information-theoretic constrained encoders Φ, modality shared 
encoder Ψ and modality complementarity estimator Λ. The information-theoretic constrained encoders Φ consist of three fully connected 
layers under the constraint of information bottleneck, which are built to remove the redundant information in each modality. 
Specifically, we feed a pair of modalities in to each Φ, in which each modality can be treated as a directive for eliminating the 
redundant information in another modality. The modality shared encoder Ψ is composed of two shared fully connected layers, which are 
constructed to translate the representations of various modalities into a shared subspace. Finally, the modality complementarity 
estimator Λ evaluates the complementary information by computing the conditional mutual information of clustering results C and 
compressed features Z from each modality. Besides, we utilize a clustering layer to generate the clustering assignment by the shared subspace.


## 2.Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.1.0
tqdm==4.48.0

## 3.Datasets

The MOSI datasets are placed in "data" folder. The others dataset we will public later. Also can find on the Internet.We will not put the link here

## 4.Usage

The pre-trained:
Concerning visual representation, we opt for the penultimate layer of VGG16 (Simonyan and Zisserman 2014) to serve 
as the 4096-dimensional features for the video frames. As for textual representation, we utilize BERT (Devlin et al.
2018) for extracting 768-dimensional semantic features. For the acoustic representation, we use a pre-trained DAVEnet 
model (Harwath et al. 2018) to extract audio features.
      Note that the data we provide has been reduced to 60.

The code includes:

- an example for train a new model：

```bash
python main.py
```



You can get the following output:

 96%|█████████▋| 482/500 [00:07<00:00, 77.42it/s] epoch 480 loss1 -1.439 loss2 0.013 loss3 -1.674 acc 0.432 nmi 0.226
 98%|█████████▊| 491/500 [00:07<00:00, 78.51it/s] epoch 490 loss1 -1.382 loss2 0.013 loss3 -1.674 acc 0.432 nmi 0.226

## 5.Experiment Results
as shown in the paper.
