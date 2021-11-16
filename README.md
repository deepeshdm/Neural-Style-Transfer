# Neural Style Transfer
Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Popular use cases for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs.

















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
