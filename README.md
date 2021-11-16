# Neural Style Transfer
### Creating Art with the help of Artificial Intelligence !

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Popular use cases for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs.

<br/> <!-- line break -->

<div align="center">
<img src="/Imgs/nst.png" width="95%"/>
</div>

<br/> <!-- line break -->


## Objective 
The main goal of this project is to explore Neural-style-transfer through implementation. We'll Implement a NST model using keras and at the end of the project we'll deploy it as a web app so that anyone can create stunning digital art which they could even sell as NFT's.


## Summary of Neural Style Transfer

Style transfer is a computer vision technique that takes two images — a "content image" and "style image" — and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image. Training a style transfer model requires two networks,which follow a encoder-decoder architecture : 
- A pre-trained feature extractor 
- A transfer network

<div align="center">
<img src="/Imgs/nst architecture.jpg" width="75%"/>
</div>



The ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.



<div align="center">
<img src="/Imgs/final_oss.png" width="45%" />
</div>


In simple words,we optimize our NST models to reduce the 'content loss' and the 'style loss'. The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image. The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image.









```python
import matplotlib.pylab as plt
from API import transfer_style

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
