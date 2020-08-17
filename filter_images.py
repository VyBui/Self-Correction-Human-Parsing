import os
import cv2


images_path = "/media/vybt/DATA1/SmartFashion/s3_data/standard_pose_img/100000_images"
part1_path = '/media/vybt/DATA1/SmartFashion/s3_data/standard_pose_img/fb_part1_24792_images'

list_images = os.listdir(part1_path)

for image_name in list_images:
    print(image_name)
    image_path = os.path.join(images_path, image_name.replace('.png', '.jpg'))
    print(image_path)
    try:
        print(f'remove {image_path}')
        os.remove(image_path)
    except:
        image_path = os.path.join(images_path, image_name)
        os.remove(image_path)


