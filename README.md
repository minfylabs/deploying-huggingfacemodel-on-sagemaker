# deploying-huggingfacemodel-on-sagemaker

In this tutorial, we'll deploy a fine tuned BERT Base model from HuggingFace Transformers on SageMaker. This model was fine tuned on propriety dataset and the weights was stored as model.bin, config.json files along with the tokenizer 

The model will be compiled using the HuggingFace SDK, and a HuggingFaceModel will be created using the Sagemaker-HuggingFace Library. A custom inference code is written which will be used to implement some basic custom logics and can be changed as per the requirements

After successfully compiling the model artifacts and uploading it to the s3 bucket, we will deploy it to a sagemkaer endpoint instance - ml.m5.xlarge

This notebook is tested ml.t3.medium SageMaker Notebook instance and the kernel type is conda_pytorch_310.
