Replication Instructions:

# Train on 2D image:
Put any 2D training data inside of data/targets. Uncomment the first block of code in src/main.py, replace the path to the image with your image. Run python3 src/main.py. The results will be stored in the results directory. 


# #Train using 3D images and camera positions:
Put any 3D training data inside of data/spheres/train, and ensure the json data representing the corresponding camera positions for the images is inside of data/spheres. Ensure the lines below train 3D aren’t commented out, and run python3 src/main.py. The results will be stored in the results directory.