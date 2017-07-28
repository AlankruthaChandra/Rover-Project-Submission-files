import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel

    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select = np.zeros_like(img[:,:,0])
    color_select[above_thresh] = 1
    #obstacle = ~color_thresh(img)
    # Return the binary image
    return color_select

def color_thresh_roc(img, rgb_thresh_max=(160, 160, 50),rgb_thresh_min = (60,100,0)):
    # Create an array of zeros same xy size as img, but single channel
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    equal_thresh = (img[:,:,0] > rgb_thresh_min[0]) \
                & (img[:,:,1] > rgb_thresh_min[1]) \
                & (img[:,:,2] > rgb_thresh_min[2]) \
                &(img[:,:,0] < rgb_thresh_max[0]) \
                & (img[:,:,1] < rgb_thresh_max[1]) \
                & (img[:,:,2] < rgb_thresh_max[2])
    color_select = np.zeros_like(img[:,:,0])
    color_select[equal_thresh] = 1


    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))

    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img

    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140],[301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img , source , destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    Navigable = color_thresh(warped)
    Obstacle = np.absolute(np.float32(Navigable) - 1)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = Obstacle * 255
    Rover.vision_image[:,:,2] = Navigable * 255

    # 5) Convert map image pixel values to rover-centric coords and defining world coordinates of Navigable and obstacle simultaneously
    xpix1, ypix1 = rover_coords(Navigable)
    x_pix_world1, y_pix_world1 = pix_to_world(xpix1, ypix1, Rover.pos[0],Rover.pos[1],Rover.yaw , 200, 10)

    xpix2, ypix2 = rover_coords(Obstacle)
    x_pix_world2, y_pix_world2 = pix_to_world(xpix2, ypix2, Rover.pos[0],Rover.pos[1],Rover.yaw , 200, 10)


    # 6) Update world map to display it on the right side of the screen
    Rover.worldmap[y_pix_world1 , x_pix_world1,2] +=10
    Rover.worldmap[y_pix_world2 , x_pix_world2,0] += 1
    nav_pix = Rover.worldmap[:, :, 2] > 0
    Rover.worldmap[nav_pix, 0] = 0

    # Finding the navigable distances and angles
    dist1, angles1 = to_polar_coords(xpix1, ypix1)
    Rover.nav_dists = dist1
    Rover.nav_angles = angles1

    # finding the rock samples and update their positions in the world map.
    Rock = color_thresh_roc(warped, rgb_thresh_max=(160, 160, 50),rgb_thresh_min = (60,100,0))
    if Rock.any():
        xpix3, ypix3 = rover_coords(Rock)
        x_pix_world3, y_pix_world3 = pix_to_world(xpix3, ypix3, Rover.pos[0],Rover.pos[1] ,Rover.yaw , 200, 10)
        dist3, angle3 = to_polar_coords(xpix3, ypix3)
        #The first occurance of rock distance is returned to the map in x and y coordinates of world map
        rock_focc = np.argmin(dist3)
        rock_xpix = x_pix_world3[rock_focc]
        rock_ypix = y_pix_world3[rock_focc]
        #Rover.worldmap[y_pix_world3, x_pix_world3]
        Rover.worldmap[rock_ypix, rock_xpix, 1] = 255
        Rover.vision_image[:,:,1] = Rock * 255
    else:
        Rover.vision_image[:,:,1] = 0





    return Rover
