## Project: Search and Sample Return

###Deepak Trivedi
### 11 October 2017

![alt text][image1]
---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./rover_image.jpg
[image2]: ./example_grid1.jpg
[image3]: ./example_rock1.jpg 
[image4]: ./threshold.png
[image5]: ./test_mapping.mp4
[image6]: ./Video_001.1_2.avi


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

The following functions were modified to allow for color selection of obstacles and rock sample:


```python
	def color_thresh(img, rgb_thresh=(160, 160, 160),flag = True):

    # Create an array of zeros same xy size as img, but single channel color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    if flag: 
        color_select[above_thresh] = 1
    else:
        color_select[~above_thresh] = 1
    # Return the binary image
    return color_select

```

A ```flag``` parameter was added to the ```color_thresh``` function to distinguish navigable ground from everything else. When ```flag``` is 1, we are looking for navigable ground. When ```flag``` is 0, we are looking for obstacles. The threshold color (160,160,160) was arrived at during the exercises in the lecture.

A second function, ```color_thresh_rock``` was created to find rocks. 

```python
def color_thresh_rock(img, rgb_thresh=(150, 100, 50)):

    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

A notable difference between this function and ```color_thresh``` is that the threshold for the blue channel has a upper threshold, instead of being a lower threshold. It is possible to merge ```color_thresh``` and ```color_thresh_rock``` into a single function by providing both lower and upper bounds. 



#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

The ```process_image()``` function was modified to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap. The following code snippet shows the modifications that were done. The function reads images from listed in the log, and for each image, applies the perspective transform, uses functions ```color_thresh``` and ```color_thresh_rock``` to identify navigable terrain, obstacles and rocks, and then applies a coordinate transform to convert the data to rover-centric and world coordinates.  Finally these are appended to the world map. 

The video output is shown in the file ```test_mapping.mp4```

[Video of the mapping step][image5] 


```python

def process_image(img):

    # Example of how to use the Databucket() object defined above

    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])
    # TODO: 
    # 1) Define source and destination points for perspective transform
    #image = mpimg.imread(img)
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])


    # 2) Apply perspective transform
    
    warped = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
    obstacles = color_thresh(warped,flag=False)
    rock_samples = color_thresh_rock(warped)
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    x_rover_navigable, y_rover_navigable = rover_coords(navigable)
    x_rover_obstacle, y_rover_obstacle = rover_coords(obstacles)
    x_rover_rock, y_rover_rock = rover_coords(rock_samples)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)

    # 5) Convert rover-centric pixel values to world coords
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    scale = 10
    world_size = data.worldmap.shape[1]
    x_world_navigable,y_world_navigable = pix_to_world(x_rover_navigable, y_rover_navigable, xpos, ypos, yaw, world_size, scale)
    x_world_obstacle,y_world_obstacle = pix_to_world(x_rover_obstacle, y_rover_obstacle, xpos, ypos, yaw, world_size, scale)
    x_world_rock,y_world_rock = pix_to_world(x_rover_rock, y_rover_rock, xpos, ypos, yaw, world_size, scale)
    # 6) Update worldmap (to be displayed on right side of screen)
    data.worldmap[y_world_obstacle, x_world_obstacle, 0] += 1
    data.worldmap[y_world_rock, x_world_rock, 1] += 0
    data.worldmap[y_world_navigable, x_world_navigable, 2] += 1

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Rover World Map vs. Ground Truth", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
``` 

	

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

The ```perception_step``` function was modified as shown below.
 
 * Source and destination points for perspective transform were identified as in the classroom exercise and the notbook. 
 * The function ```perspect_transform``` was used to apply perspective transform
 * The ```color_thresh``` and ```color_thresh_rock``` were used to identify navigable terrain/obstacles/rock samples
 *   Updated ```Rover.vision_image``` for the output to be shown on the screen
 *   Apply coordinate transformations to rover-centric and world coordinates and update worldmap 
 *   Convert pixels to polar coordinates and use these to update the ```Rover``` data structure.

```python

def perception_step(Rover):

    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    image =Rover.img
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
    obstacles = color_thresh(warped,flag=False)
    rock_samples = color_thresh_rock(warped)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacles
    Rover.vision_image[:,:,1] = rock_samples
    Rover.vision_image[:,:,2] = navigable

    # 5) Convert map image pixel values to rover-centric coords
    x_rover_navigable, y_rover_navigable = rover_coords(navigable)
    x_rover_obstacle, y_rover_obstacle = rover_coords(obstacles)
    x_rover_rock, y_rover_rock = rover_coords(rock_samples)

    # 6) Convert rover-centric pixel values to world coordinates
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    scale = 10
    world_size = Rover.worldmap.shape[1]
    x_world_navigable,y_world_navigable = pix_to_world(x_rover_navigable, y_rover_navigable, xpos, ypos, yaw, world_size, scale)
    x_world_obstacle,y_world_obstacle = pix_to_world(x_rover_obstacle, y_rover_obstacle, xpos, ypos, yaw, world_size, scale)
    x_world_rock,y_world_rock = pix_to_world(x_rover_rock, y_rover_rock, xpos, ypos, yaw, world_size, scale)
    
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world_obstacle, x_world_obstacle, 0] += 1
    Rover.worldmap[y_world_rock, x_world_rock, 1] += 1
    Rover.worldmap[y_world_navigable, x_world_navigable, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(x_rover_navigable, y_rover_navigable)
    mean_dir = np.mean(angles)
    
    # Update Rover pixel distances and angles
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    
 
    
    
    return Rover

```

The figure below shows an example of the application of `process_image()`. The image recorded by the rover is on the top left. The image after the projective transformation is on the top-right. The result of `color_thresh` is shown on bottom left, where black indicates obstacle and white indicates navigable terrain. Finally, on the bottom right, the red arrow indicates the vector joining the centroid of the pixels in polar coordinates to the rover.   

![alt text][image4] 

Only minor changes were required in ```decision_step(Rover)``` . Different values of `Rover.steer` clipping angle and minimum velocity `Rover.vel` were experimented with. A maximum steering angle of 20 degrees and a minimum velocity to determine stopping of 0.5 m/s were finally chosen. Please look at ```decision.py``` for these details.  

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

The script `drive_rover.py` was launched along with the simulator to run the rover autonomously. Several autonomous runs were made, with different rover parameters. The following parameters were experimented with in the file `drive_rover.py`: 

* `self.max_vel` in the range 2 to 4. 
* `self.brake_set` in the range 5 to 20
* `self.throttle_set` in the range 0.1 to 0.3

These experiments were done just for understanding how the rover performance changes with its dynamics. 

A video of the rover simulation is provided in the file `Video_001.1_2.avi`. The video shows that the rover successfully maps at least 40% of the environment with 60% fidelity (accuracy) against the ground truth. It also finds (maps) the location of at least one rock sample. In fact, in this video, the rover maps 48% of the environment with 66% fidelity and locates two rock samples. Other simulations have shown varying results, with mapping up to 100%, fidelity up to 70% and locating up to 5 samples. However, in some cases, the rover has also shown the tendency to get stuck on a rock in the middle of navigable terrain, and also in a few cases, entering an infinite loop, circling around the same track. 

[Video of the simulation][image6]	  

For this simulation, the screen resolution used was `720x480`, the the graphics quality was `Good`. The output was generated at 22 frames per second.

Following are some approaches that could be used to improve the algorithm further:

1. One major drawback of my approach is that it does not mark the visited terrain. Nothing prevents it from entering an infinite loop and keep revisiting the same terrain over and over again, while not visiting other parts of the world. An easy way to implement this in `decision_step` could be to store all the `visited` pixels of the world map, and calculate centroid of those pixels. Given a decision point where a direction of movement needs to be chosen, minimize the dot product of possible directions with respect to the vector joining the rover to the centroid. 

2. I did not complete the optional exercise of collecting samples and returning these to the starting point, since I ran out of time. I would like to take that up at a later date. 
3. Clearly, many of the heuristics used in this exercise are impractical in real scenarios, such as using simple color thresholds for recognizing navigable land and rock samples. A practical implementation will require a more sophisticated approach, such as a neural network trained on many rock samples. 


**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines! 





