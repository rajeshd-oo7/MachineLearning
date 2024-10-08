In this project we build a classifier based on Rajesh Dachiraju's paper (arxiv, [cited below](https://arxiv.org/abs/2011.11258)). The problem in using the method described in this paper is that the number of computations grow exponentially with the increase in feature dimensions. According to the paper "Testing the Manifold Hypothesis", every high dimensional data lies in the vicinity of a low dimensional manifold. So to overcome the high computations problem, before using the method, we do a non linear dimensionality reduction. The method we choose is UMAP.
The classifier we build in this project is to classify color fundus images to determine Diabetic Retinopathy disease. (DR or no DR). Image size is 244x244x3. Dataset is from Kaggle datasets 2019. (URL: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data ).
Download the dataset from the given URL and place it in the data folder and run the 'data_prep_DiabeticRetinopathy.py' script (update the data path in the script). The script extracts the image data and perfoms nonlinear dimensionality reduction using UMAP.
It reduces 244x244x3 =  1,50,528 dimensions to 4 dimensions. The final feature vectors are 4 dimensional. The feture vector and data label files are generated.  Then run the 'DR_classifier_python.py' classifier script.

Note: alpha_sens, beta_sens, omega_scale_factor, lambda_scale_factor are the adjustable parameters for any new dataset. 

Note: The classifier is trained and run iteratively using point selection, instead of drawing random points. For low accuracy problems and other purposes one should modify code to not use point selection and just draw random points without any iterations.


References

1. Dachiraju, R., 2020. Approximation of a Multivariate Function of Bounded Variation from its Scattered Data. arXiv preprint arXiv:2011.11258.

2. Testing the manifold hypothesis by Charles Fefferman, Sanjoy Mitter and Hariharan Narayanan J. Amer. Math. Soc. 29 (2016), 983-1049 DOI: https://doi.org/10.1090/jams/852.

3. McInnes, L., Healy, J. and Melville, J., 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

4. The code cartesian product is due to this post at this url: https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/49445693#49445693

5. The function/code Computational fill distance is based on the post here : https://mathoverflow.net/q/390461/134538
