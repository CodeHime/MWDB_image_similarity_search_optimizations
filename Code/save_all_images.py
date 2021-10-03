# Save all arrays from dataset as greyscale images
# Using PIL as matplotlib takes > 5min to save and PIL < 30 seconds
"""
Matplotlib alternative code:
# plt.axis('off')  
# imgplot=plt.imshow(img_data["images"][i], cmap='gray')
# plt.savefig(input_dir + f"{img_data['target'][i]}/{i%10}.png", bbox_inches='tight', pad_inches=0)
# plt.axis('on') 
"""
# Note: PIL is smart so if u ran and deleted these images before, make sure to empty your trash before running the same code again
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import os

def save_all_images(input_dir, img_data):   
  if not os.path.isdir(input_dir):
    os.makedirs(input_dir)

  # Save all input images
  for i in range(img_data["images"].shape[0]):
    # if not os.path.isdir(input_dir + f"{img_data['target'][i]}/"):
    #   os.makedirs(input_dir + f"{img_data['target'][i]}/")

    result = Image.fromarray((img_data["images"][i]* 255).astype(np.uint8))
    # result.save(input_dir + f"{img_data['target'][i]}/{i%10}.png") 
    result.save(input_dir + f"{i}.png") 


base_dir= "./"
image_dir=base_dir + "input_images/"
img_data = fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
# img_data.keys()
save_all_images(input_dir, img_data)