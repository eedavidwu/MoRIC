import os
import shutil
from PIL import Image
size=64
file_dir = './dataset/kodak/kodak_data/'
out_floder='./dataset/kodak_'+str(size)+'/'
os.mkdir(out_floder)

file_name_list = os.listdir(file_dir)
for file_name_full in file_name_list:
    file_name=os.path.splitext(file_name_full)[0]
    out_data_folder=os.path.join(out_floder,file_name)
    os.mkdir(out_data_folder)
    out_data_folder=os.path.join(out_data_folder,'data')
    os.mkdir(out_data_folder)


    file_path = os.path.join(file_dir,file_name_full)
    im = Image.open(file_path)
    #print(im.size)
    for i in range (im.size[0]//size):
        for j in range (im.size[1]//size):
            #box=(i*256,j*256,(i+1)*256,(j+1)*256)
            box=(i*size,j*size,(i+1)*size,(j+1)*size)

            img_patch=im.crop(box)
            save_img_path=os.path.join(out_data_folder,file_name+'_'+str(j)+'_'+str(i)+'.png')
            print(save_img_path)
            img_patch.save(save_img_path)
        

