# EXPLAINABLE-AI-AND-ATTENTION-DRIVEN-MODELS-FOR-BRAIN-TUMOR-SEGMENTATION
## Dataset
The dataset contains around 2200 images containing two classes, Tumor(1)/No tumor(0), split into 70% training, 20% validation, and 10% testing.
Dataset link: [https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data](url)

## Method
Our method integrates advanced techniques, such as saliency maps, attention layers, and morphological operations, to enhance the performance of the U-Net model for brain tumor classification and segmentation. These additions improve feature localization, model interpretability, and segmentation accuracy.

 + Saliency Maps: Saliency maps highlight the most critical regions of the input image that influence the model’s predictions. By incorporating saliency maps into the training process,
 we emphasize the areas of interest, such as tumor boundaries, ensuring that the model learns features most relevant to the task.
 + Attention Layers: We enhance the U-Net architecture by adding attention mechanisms. These layers adaptively focus on the most informative parts of the feature maps during training, reducing noise from irrelevant regions. This improves the model’s ability to capture fine details in tumor boundaries while retaining global contextual information.
 + Morphological Operations: Post-processing with morphological operations, such as dilation and erosion, refines the segmentation masks. These operations help to smooth edges, remove noise, and fill gaps in the predicted tumor regions, leading to cleaner and more clinically helpful segmentation results.

