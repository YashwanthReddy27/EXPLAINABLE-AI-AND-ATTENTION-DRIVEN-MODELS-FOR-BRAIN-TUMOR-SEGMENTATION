# EXPLAINABLE-AI-AND-ATTENTION-DRIVEN-MODELS-FOR-BRAIN-TUMOR-SEGMENTATION
## Dataset
The dataset contains around 2200 images containing two classes, Tumor(1)/No tumor(0), split into 70% training, 20% validation, and 10% testing.
Dataset link: [https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data](url)

## Method
<p style="text-align: justify;">
Our method integrates advanced techniques, such as saliency maps, attention layers, and morphological operations, to enhance the performance of the U-Net model for brain tumor classification and segmentation. These additions improve feature localization, model interpretability, and segmentation accuracy.

+ **Saliency Maps**: Saliency maps highlight the most critical regions of the input image that influence the model’s predictions. By incorporating saliency maps into the training process, we emphasize the areas of interest, such as tumor boundaries, ensuring that the model learns features most relevant to the task.
+ **Attention Layers**: Adding attention mechanisms enhances the U-Net architecture. These layers adaptively focus on the most informative parts of the feature maps during training, reducing noise from irrelevant regions. This improves the model’s ability to capture fine details in tumor boundaries while retaining global contextual information.
+ **Morphological Operations**: Post-processing with morphological operations, such as dilation and erosion, refines the segmentation masks. These operations help to smooth edges, remove noise, and fill gaps in the predicted tumor regions, leading to cleaner and more clinically helpful segmentation results.

The attention mechanism enhances features from the encoder for the decoder using spatial attention. Skip connections and gating signals from the decoder are reduced in dimensions through 1x1 convolutions, combined, passed through ReLU, and processed using sigmoid activation to create a spatial mask. This mask refines the skip connection, improving segmentation by focusing on critical regions.

+ **Improved Localization**: Attention gates enhance relevant regions and ignore background noise, improving segmentation accuracy.
+ **Better Generalization**: Emphasizing important features helps the model generalize and avoid overfitting.

Spatial attention mechanisms assign higher importance to informative pixels, improving boundary precision in tasks like tumor segmentation. Channel attention mechanisms amplify relevant feature maps, leading to robust tumor representation.

Morphological operations refine segmentation by smoothing edges, connecting regions, filling holes, and enhancing boundary precision, improving metrics like Dice Coefficient and IoU.
</p>
### Explainable AI (XAI)
XAI enhances interpretability using saliency maps, highlighting influential areas of an input image.

### Built Models
1. **U-Net**
   The U-Net model with attention gates is a sophisticated architecture for medical image segmentation tasks, such as brain tumor detection. It features an encoder (contracting path) and a decoder (expanding path). The encoder progressively down-samples the input image through convolutional layers and max-pooling, employing dropout layers to prevent overfitting. Each encoder block contains two 3x3 convolutional layers activated by ReLU. Two 3x3 convolutional layers with 1024 filters capture the most complex features at the bottleneck. In the decoder, the model up-samples feature maps using transposed convolution layers, doubling their spatial dimensions at each step. Attention gates within the decoder dynamically weight the encoder feature before concatenation, which helps highlight relevant areas and suppress irrelevant ones, thus improving segmentation accuracy. Each decoder block includes an attention gate and two 3x3 convolutional layers to refine the feature maps. The output layer is a 1x1 convolution with a sigmoid activation, generating a binary mask for precise segmentation. This attention mechanism significantly enhances the model’s ability to focus on essential structures, making it particularly useful for high-accuracy medical image analysis.


