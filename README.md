
### 🎨🖌 Creating Art with the help of Artificial Intelligence !

**🔥 Official Website :** https://share.streamlit.io/deepeshdm/pixelmix/main/App.py

<div align="center">
  <img src="/Imgs/website.gif" width="90%"/>
</div>
</br>

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Popular use cases for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs.

<br> <!-- line break -->

<div align="center">
<img src="/Imgs/nst.png"/>
</div>

<br> <!-- line break -->


## 🎯 Objective 
The main goal of this project is to explore Neural-style-transfer through implementation. We'll Implement a NST model using Tensorflow and keras, and at the end of the project we'll deploy it as a web app so that anyone can create stunning digital art which they could even sell as NFT's.


## 📝 Summary of Neural Style Transfer

Style transfer is a computer vision technique that takes two images — a "content image" and "style image" — and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image. Training a style transfer model requires two networks,which follow a encoder-decoder architecture : 
- A pre-trained feature extractor 
- A transfer network


<div align="center">
<img src="/Imgs/nst architecture.jpg" width="80%"/>
</div>

<br> <!-- line break -->



The ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.



<div align="center">
<img src="/Imgs/final_oss.png" width="50%" />
</div>

<br> <!-- line break -->


In simple words,we optimize our NST models to reduce the 'content loss' and the 'style loss'. The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image. The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image.


## 👨‍💻 Implementation

Early versions of NST treated the task as an optimization problem, requiring hundreds or thousands of iterations to perform style transfer on a single image. To tackle this inefficiency, researchers developed what’s referred to as "Fast Neural Style Transfer". Fast style transfer also uses deep neural networks but trains a standalone model to transform any image in a single, feed-forward pass. Trained models can stylize any image with just one iteration through the network, rather than thousands.State-of-the-art style transfer models can even learn to imprint multiple styles via the same model so that a single input content image can be edited in any number of creative ways.

In this project we used a pre-trained "Arbitrary Neural Artistic Stylization Network" - a Fast-NST architecture which you can find [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2). The model is successfully trained on a corpus of roughly 80,000 paintings and is able to generalize to paintings previously unobserved.


## To run locally

1. Download the pre-trained TF model.

    - The 'model' directory already contains the pre-trained model,but you can also download the pre-trained model from [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

2. Import this repository using git command
```
git clone https://github.com/deepeshdm/Neural-Style-Transfer.git
```
3. Install all the required dependencies inside a virtual environment
```
pip install -r requirements.txt
```
4. Copy the below code snippet and pass the required variable values
```python
import matplotlib.pylab as plt
from API import transfer_style

# Path of the downloaded pre-trained model or 'model' directory
model_path = r"C:\Users\Desktop\magenta_arbitrary-image-stylization-v1-256_2"

# NOTE : Works only for '.jpg' and '.png' extensions,other formats may give error
content_image_path = r"C:\Users\Pictures\my_pic.jpg"
style_image_path = r"C:\Users\Desktop\images\mona-lisa.jpg"

img = transfer_style(content_image_path,style_image_path,model_path)
# Saving the generated image
plt.imsave('stylized_image.jpeg',img)
plt.imshow(img)
plt.show()
```

## 🔥 Web Interface & API

In order to make it easy for anyone to interact with the model,we created a clean web interface using Streamlit and deployed it on their official cloud space.

- Checkout Official Website : https://share.streamlit.io/deepeshdm/pixelmix/main/App.py
- Website Repository : [here](https://github.com/deepeshdm/PixelMix)

<div align="center">
  <img src="/Imgs/website.gif" width="90%"/>
</div>


## 🖼🖌 Some of the art we created in this project

<div align="center">
  <img src="/Imgs/content1.jpg" width="35%"/>
<img src="/Imgs/art1.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content2.jpg" width="35%"/>
<img src="/Imgs/art2.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content3.jpg" width="35%"/>
<img src="/Imgs/art3.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content4.jpg" width="35%"/>
<img src="/Imgs/art4.png" width="35%"/>
</div>

References :
- https://arxiv.org/abs/1508.06576 
- https://keras.io/examples/generative/neural_style_transfer/ 
- https://arxiv.org/abs/1705.06830 
- https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2 















