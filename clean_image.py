from os import listdir

from PIL import Image, ImageFilter


# Parameters
# Input directory of images
input_dir = "media/volibear/voli_q"
# Output directory of masked and cropped images
output_dir = "media/voli-q-cleaned"
# Area to pre-crop the images to (min_x, min_y, max_x, max_y), can save runtime for large screenshots with small objects
# Teemo model viewer:
area = (700, 170, 1768, 928)
# Teemo model viewer pink background
# purple background
background = (95, 80, 170)
# Teemo model viewer tolerance (25)
tolerance = 19
# This is needed because there is another shade of pink in the background
tolerance_offset_1 = 1.0
tolerance_offset_2 = 1.0  # Greenscreen: 0.74
tolerance_offset_3 = 2.5  # Teemo viewer: 2.5
# Remove the outline of the images in case there are any masking artifacts, set the number of layers to be removed
remove_outline = 0
#####################################################
"""
This function applies a filter to a masked image and can either remove 
or add differntly colored borders around objects
"""


def modify_outline(file_name, thickness):
    image = Image.open(output_dir + "/" + file_name + ".png")
    image = image.convert("RGBA")
    for t in range(thickness):
        mask = image.filter(ImageFilter.FIND_EDGES)
        mask_data = mask.getdata()
        image_data = image.getdata()
        w, h = mask_data.size

        out_data = []
        for y in range(0, h):
            for x in range(0, w):
                index = x + w * y
                pixel = (0, 0, 0, 0)
                if mask_data[index][3] > 0:
                    pixel = (255, 255, 255, 0)
                else:
                    pixel = (image_data[index][0], image_data[index][1], image_data[index][2], image_data[index][3])
                out_data.append(pixel)
        image.putdata(out_data)
    image.save(output_dir + "/" + file_name + ".png", "PNG")


def get_min_max_x(new_data, w, h):
    min_value = 0
    max_value = 0
    for x in range(w - 1, 0, -1):
        for y in range(0, h - 1):
            data_index = x + w * y
            if new_data[data_index][3] != 0:
                max_value = x
                break
        else:
            continue
        break
    for x in range(0, w - 1):
        for y in range(0, h - 1):
            data_index = x + w * y
            if new_data[data_index][3] != 0:
                min_value = x
                break
        else:
            continue
        break
    return min_value, max_value


def get_min_max(new_data, w, h):
    min_value = 0
    max_value = 0
    for y in range(h - 1, 0, -1):
        for x in range(0, w - 1):
            data_index = x + w * y
            if new_data[data_index][3] != 0:
                max_value = y
                break
        else:
            continue
        break
    for y in range(0, h - 1):
        for x in range(0, w - 1):
            data_index = x + w * y
            if new_data[data_index][3] != 0:
                min_value = y
                break
        else:
            continue
        break
    return min_value, max_value


# Get list of files in the input directory
files = sorted(listdir(input_dir))

for f in files:
    # Remove the jpg ending
    # temp = "."
    # fname = f.split(".")[0] + temp + f.split(".")[1]
    # print("processing:", fname)
    # img = Image.open(input_dir + "/" + fname + ".jpg")
    fname = f.split(".")[0]
    print(fname)
    img = Image.open(input_dir + "/" + fname + ".jpg")
    # Add alpha channel
    img = img.convert("RGBA")

    # Pre crop to save runtime
    cropped = img.crop(area)
    datas = cropped.getdata()
    newData = []
    for item in datas:
        if item[0] > background[0] - tolerance_offset_1 * tolerance and item[0] < background[
            0] + tolerance_offset_1 * tolerance \
                and item[1] > background[1] - tolerance * tolerance_offset_2 and item[1] < background[
            1] + tolerance_offset_2 * tolerance \
                and item[2] > background[2] - tolerance_offset_3 * tolerance and item[2] < background[
            2] + tolerance_offset_3 * tolerance:
            newData.append((255, 255, 255, 0))
        else:
            newData.append((item[0], item[1], item[2], 255))

    # Save new image data
    cropped.putdata(newData)
    w, h = cropped.size  # Crop image to pixel content
    min_y, max_y = get_min_max(newData, w, h)
    # Trick: rotate the image by 9 degrees and apply the same functions
    temp = cropped.rotate(90)
    temp_w, temp_h = temp.size
    tempData = temp.getdata()
    min_x, max_x = get_min_max_x(newData, w, h)
    # Save output image as png
    cropped = cropped.crop((min_x, min_y, max_x, max_y))
    cropped.save(output_dir + "/" + fname + ".png", "PNG")
    # Remove the outer layer of pixels to remove artifacts from cropping
    modify_outline(fname, remove_outline)
