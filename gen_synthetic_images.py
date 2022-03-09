# Copyright 2019 Oliver Struckmeier
# Licensed under the GNU General Public License, version 3.0. See LICENSE for details

import numpy as np
from PIL import Image
from PIL import ImageFilter
from os import listdir
import random

# Object Classes
# 0 - idle
# 1 - auto attack
# 2 - q ability
# 3 - w ability
# 4 - e ability
# 5 - r ability

# Params #
# Print out the status messages and where stuff is placed
debug = True

# Directory of the cleaned/masked images
masked_images_dir = "media/voli-cleaned"

cleaned_aa_dir = "media/voli-aa-cleaned"
cleaned_q_dir = "media/voli-q-cleaned"
cleaned_w_dir = "media/voli-w-cleaned"
cleaned_e_dir = "media/voli-e-cleaned"
cleaned_r_dir = "media/voli-r-cleaned"

# Directory in which the map backgrounds are located
map_images_dir = "media/map"
# Directory in which the map backgrounds with fog of war are located
map_fog_dir = "media/map-fog"
# Directory for output
output_dir = "train_data"
# Sometimes randomly add the overlay
overlay_chance = 25
overlay_path = "ui"

# Prints a bounding box around the placed object in red (for debug purposes)
print_box = False

# Size of the datasets the program should generate
dataset_size = 250

# Beginning index for naming output files
start_index = 0

# How many characters should be added minimum/maximum to each sample
characters_min = 0
characters_max = 1
assert (characters_min <= characters_max), "Error, characters_max needs to be larger than character_min!"

# The scale factor of how much a champion image needs to be scaled to have a realistic size
# Also you can set a random factor to create more diverse images
scale_champions = 0.7  # 0.7 good
random_scale_champions = 0.12  # 0.12 is good

# Random rotation maximum offset in counter-/clockwise direction
rotate = 10

# Make champions semi-transparent sometimes to simulate them being in a brush/invisible, value in percent chance a
# champion will be semi-transparent
invisibility_prob = 22

# Output image size
output_size = (1920, 1080)

# Factor how close the objects should be clustered around the bias point larger->less clustered but also more out of
# the image bounds, value around 100 -> very clustered
bias_strength = 220  # 220 is good, don't select too large or the objects will be too often out of bounds

# Resampling method of the object scaling
# sampling_method = Image.BICUBIC
sampling_method = Image.BILINEAR  # IMO the best but use both to have more different methods

# Add random noise to pixels, value in percent chance in which image will have noise applied
noise_prob = 23
noise = (40, 40, 40)

# Blur the image
blur = False
blur_strength = 0.8  # 0.6 is a good value

# Probability of adding a fog of war screenshot with no objects in it
fog_of_war_prob = 8

# Padding for the bias point (to keep the clustering of the minions from spawning minions outside of the image
padding = 300  # 400 is good

# Helper functions #
"""
This funciton applies random noise to the rgb values of a pixel (R,G,B)
"""


def apply_noise(pixel):
    r = max(0, min(255, pixel[0] + random.randint(-noise[0], noise[0])))
    g = max(0, min(255, pixel[1] + random.randint(-noise[1], noise[1])))
    b = max(0, min(255, pixel[2] + random.randint(-noise[2], noise[2])))
    a = pixel[3]
    return r, g, b, a


"""
This function places a masked image with a given path onto a map fragment
Passing -1 to the object class allows you to set objects like the UI that are not affected by rotations bias etc.
"""


def add_object(path, cur_image_path, object_class, bias_point, last):
    # Set up the map data
    map_image = Image.open(cur_image_path)
    map_image = map_image.convert("RGBA")
    # Cut the image to the desired output image size
    map_data = map_image.getdata()
    w, h = map_image.size

    # if debug:
    #     print("Adding object: ", path)

    # Read the image file of the current object to add
    obj = Image.open(path)

    # Randomly rotate the image, but make the normal orientation most likely using a normal distribution
    if object_class >= 0:
        obj = obj.rotate(np.random.normal(loc=0.0, scale=rotate), expand=True)
    obj = obj.convert("RGBA")
    obj_w, obj_h = obj.size

    # Rescale the image based on the scale factor
    if object_class >= 0:  # e.g. a champion's class identifier
        scale_factor = random.uniform(scale_champions - random_scale_champions,
                                      scale_champions + random_scale_champions)
        size = int(obj_w * scale_factor), int(obj_h * scale_factor)
    else:
        size = int(obj_w), int(obj_h)

    # size = int(obj_w), int(obj_h)

    # If the object is a champion make it semi-transparent sometimes to simulate it being in a brush/invisible
    in_brush = False
    if object_class >= 0 and np.random.randint(0, 100) > 100 - invisibility_prob:
        in_brush = True

    # Compute the position of minions based on the bias point. Normally distribute the minions around
    # a central point to create clusters of objects for more realistic screenshot fakes
    # Champions and structures are uniformly distributed
    if object_class >= 0:  # Champion
        obj_pos_center = (random.randint(0, w - 1), random.randint(0, h - 1))
    else:
        x_coord = np.random.normal(loc=bias_point[0], scale=bias_strength)
        y_coord = np.random.normal(loc=bias_point[1], scale=bias_strength)
        obj_pos_center = (int(x_coord), int(y_coord))

    # Catch the -1 object class exception to add for example the overlay that has to be centered
    if object_class == -1:  # overlay
        obj_pos_center = (int(w/2), int(h/2))

    # Resize the image based on the scaling above
    obj = obj.resize(size, resample=sampling_method)
    obj_w, obj_h = obj.size

    if debug:
        print("Placing at : [{},{}] with {} transparency".format(obj_pos_center[0], obj_pos_center[1], in_brush))

    # Extract the image data
    obj_data = obj.getdata()
    out_data = np.array(map_image)
    last_pixel = 0

    # Compute the object corners
    min_x = int(min(w, max(0, obj_pos_center[0] - obj_w / 2 - 2)))
    max_x = int(min(w, max(0, obj_pos_center[0] + obj_w / 2 + 2)))
    min_y = int(min(h, max(0, obj_pos_center[1] - obj_h / 2 - 2)))
    max_y = int(min(h, max(0, obj_pos_center[1] + obj_h / 2 + 2)))

    # Place the images
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            pixel = (0, 0, 0, 0)
            # Compute the pixel index in the map fragment
            map_index = x + w * y
            # If we want to print the box around the object, set the pixel to red
            if print_box is True and y == obj_pos_center[1] - int(obj_h / 2) and \
                    obj_pos_center[0] - int(obj_w / 2) < x < obj_pos_center[0] + int(obj_w / 2):
                pixel = (255, 0, 0, 255)
            elif print_box is True and y == obj_pos_center[1] + int(obj_h / 2) and \
                    obj_pos_center[0] - int(obj_w / 2) < x < obj_pos_center[0] + int(obj_w / 2):
                pixel = (255, 0, 0, 255)
            elif print_box is True and x == obj_pos_center[0] - int(obj_w / 2) and \
                    obj_pos_center[1] - int(obj_h / 2) < y < obj_pos_center[1] + int(obj_h / 2):
                pixel = (255, 0, 0, 255)
            elif print_box is True and x == obj_pos_center[0] + int(obj_w / 2) and \
                    obj_pos_center[1] - int(obj_h / 2) < y < obj_pos_center[1] + int(obj_h / 2):
                pixel = (255, 0, 0, 255)
            else:
                # Replace the old input image pixels with the object to add pixels
                if obj_pos_center[0] - int(obj_w / 2) <= x <= obj_pos_center[0] + int(obj_w / 2) \
                        and obj_pos_center[1] - int(obj_h / 2) <= y <= obj_pos_center[1] + int(obj_h / 2):
                    obj_x = x - obj_pos_center[0] - int(obj_w / 2) - 1
                    obj_y = y - obj_pos_center[1] - int(obj_h / 2) - 1
                    object_index = (obj_x + obj_w * obj_y)
                    # Check the alpha channel of the object to add If it is smaller 150, the pixel is invisible,
                    # 255: fully visible, 150: semi-transparent (brush simulation) Then use the original image's pixel
                    # value Else use the object to adds pixel value
                    if obj_data[object_index][3] == 255:
                        if in_brush and last_pixel % 3 == 0:
                            # take the map pixel every second time to make the champion semi-transparent
                            pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2],
                                     map_data[map_index][3])
                            last_pixel += 1
                        else:
                            pixel = (
                                obj_data[object_index][0], obj_data[object_index][1], obj_data[object_index][2], 255)
                            last_pixel += 1
                    elif obj_data[object_index][3] == 0:
                        pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], 255)
                else:
                    pixel = (
                        map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
            out_data[y, x] = pixel

    # Apply noise filter to the image
    if last and (noise[0] > 0 or noise[1] > 0 or noise[2] > 0) and np.random.randint(0, 100) > 100 - noise_prob:
        if debug:
            print("Applying noise filter")
        for y in range(0, h):
            for x in range(0, w):
                # Compute the pixel index in the map fragment
                map_index = x + w * y
                out_data[y, x] = apply_noise(out_data[y, x])

    # Saving the image
    map_image = Image.fromarray(np.array(out_data))

    # Apply blur
    if blur and last:
        map_image = map_image.filter(ImageFilter.GaussianBlur(radius=blur_strength))

    map_image = map_image.convert("RGB")
    map_image.save(output_dir + "/images/" + filename + ".jpg", "JPEG")

    # Append the bounding box data to the labels file if the object class is not -1
    if object_class >= 0:
        with open(output_dir + "/labels/" + filename + ".txt", "a") as f:
            # Write the position of the object and its bounding box data to the labels file
            # All values are relative to the whole image size
            # Format: class, x_pos, y_pos, width, height
            f.write("" + str(object_class) + " " + str(float(obj_pos_center[0] / w)) + " " + str(
                float(obj_pos_center[1] / h)) + " " + str(float(obj_w / w)) + " " + str(float(obj_h / h)) + "\n")


# Main function
obj_dirs = sorted(listdir(masked_images_dir))
aa_dirs = sorted(listdir(cleaned_aa_dir))
q_dirs = sorted(listdir(cleaned_q_dir))
w_dirs = sorted(listdir(cleaned_w_dir))
e_dirs = sorted(listdir(cleaned_e_dir))
r_dirs = sorted(listdir(cleaned_r_dir))
maps = sorted(listdir(map_images_dir))

for dataset in range(0, dataset_size):
    filename = str(dataset + start_index)
    print("Dataset: ", dataset, " / ", dataset_size, " : ", filename)
    # Randomly select a map background
    map_bg = map_images_dir + "/" + random.choice(maps)
    if debug:
        print("Using map fragment: ", map_bg)

    # Randomly select a set of characters to add to the image
    characters = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(obj_dirs)
        temp_obj_path = masked_images_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        characters.append(
            [masked_images_dir + "/" + temp_obj_folder, 0])

    if debug:
        print("Adding {} Idle class!".format(len(characters)))

    skills_aa = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(aa_dirs)
        temp_obj_path = cleaned_aa_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        skills_aa.append(
            [cleaned_aa_dir + "/" + temp_obj_folder, 1])

    if debug:
        print("Adding {} AA class)!".format(len(skills_aa)))

    skills_q = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(q_dirs)
        temp_obj_path = cleaned_q_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        skills_q.append(
            [cleaned_q_dir + "/" + temp_obj_folder, 2])

    if debug:
        print("Adding {} Q class)!".format(len(skills_q)))

    skills_w = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(w_dirs)
        temp_obj_path = cleaned_w_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        skills_w.append(
            [cleaned_w_dir + "/" + temp_obj_folder, 3])

    if debug:
        print("Adding {} W class)!".format(len(skills_w)))

    skills_e = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(e_dirs)
        temp_obj_path = cleaned_e_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        skills_e.append(
            [cleaned_e_dir + "/" + temp_obj_folder, 4])

    if debug:
        print("Adding {} E class)!".format(len(skills_w)))

    skills_r = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(r_dirs)
        temp_obj_path = cleaned_r_dir + "/" + temp_obj_folder
        # Select a random masked image of object
        skills_r.append(
            [cleaned_r_dir + "/" + temp_obj_folder, 5])

    if debug:
        print("Adding {} R class)!".format(len(skills_w)))

    # Add a fog of war / empty screenshot
    if 100 - fog_of_war_prob < random.randint(0, 100):
        fog_file = random.choice(sorted(listdir(map_fog_dir)))
        fog_name = map_fog_dir + "/" + fog_file
        map_image = Image.open(fog_name)
        map_image.save(output_dir + "/images/" + filename + ".jpg", "JPEG")
        # Save empty label because we did not place any objects
        with open(output_dir + "/labels/" + filename + ".txt", "a") as f:
            f.write("")
    else:
        # Now figure out the order in which we want to add the objects (So that sometimes objects will overlap)
        objects_to_add = characters + skills_aa + skills_q + skills_w + skills_e + skills_r
        random.shuffle(objects_to_add)
        # Read in the current map background as image
        map_image = Image.open(map_bg)
        w, h = map_image.size
        # Make sure the image is 1920x1080 (otherwise the overlay might not fit properly)
        assert (w == 1920 and h == 1080), "Error image has to be 1920x1080"

        map_image.save(output_dir + "/images/" + filename + ".jpg", "JPEG")
        cur_image_path = output_dir + "/images/" + filename + ".jpg"
        # Iterate through all objects in the order we want them to be added and add them to the background
        # Note this function also saves the image already
        # Point around which the objects will be clustered
        bias_point = (random.randint(padding, w - 1 - padding), random.randint(padding, h - 1 - padding))
        # Add the overlay, the bias point plays no role here because of the object class (object class -1 is not
        # added to the labels.txt)
        if random.randint(0, 100) > 100 - overlay_chance:
            if debug:
                print("Adding UI overlay")

            overlay_name = overlay_path + "/" + random.choice(sorted(listdir(overlay_path)))
            add_object(overlay_name, cur_image_path, -1, bias_point, False)
        for i in range(0, len(objects_to_add)):
            o = objects_to_add.pop()
            if len(objects_to_add) == 0:
                add_object(o[0], cur_image_path, o[1], bias_point, True)  # Set last to true to apply the possible noise
            else:
                add_object(o[0], cur_image_path, o[1], bias_point, False)
        with open(output_dir + "/labels/" + filename + ".txt", "a") as f:
            f.write("")
    if debug:
        print("=======================================")
