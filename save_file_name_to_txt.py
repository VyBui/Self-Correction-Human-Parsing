import os


list_files = os.listdir('/media/vybt/DATA1/SmartFashion/s3_data/standard_pose_img/schp/val_images')

file_txt = open('/media/vybt/DATA1/SmartFashion/s3_data/standard_pose_img/schp/val_id.txt',
                    'w')

for image_name in list_files:
    image_name = image_name.split('.')[0]
    file_txt.write(image_name + "\n")
file_txt.close
