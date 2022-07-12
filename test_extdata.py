from logging import raiseExceptions
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from  src.utils import *
initial_path=os.getcwd()
from src.visualization.visualize import read_image,show_raw_images, show_images, save_images
from src.data.cp_dataset import CPDataset
from src.models.networks import GMM, UnetGenerator,load_checkpoint
import warnings
warnings.filterwarnings("ignore")
from src.data.build_image_mask import make_body_mask, make_cloth_mask
from openpose_api.keypoints_from_images import find_keypoints
from graphonomy.inference import inference
import time
import shutil

#initial folder where unprocessed images are saved
RAW_FOLDER = Path('data/raw')

GRAPHONOMY_MODEL_PATH ="checkpoints\graphonomy\inference.pth"



def process_images(image_names,args, processed_folder):
    """
    Process images and save them in processed_folder
    
    args:
        image_names: list of (image_path,cloth_path)
        args: args
        processed_folder: path where processed images are saved
    """
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    processed_image_dir= os.path.join(processed_folder, "image")
    if not os.path.exists(processed_image_dir):
        os.makedirs(processed_image_dir)

    #create folder data/processed/cloth
    processed_cloth_dir = os.path.join(processed_folder, "cloth")
    if not os.path.exists(processed_cloth_dir):
        os.makedirs(processed_cloth_dir)

    processed_image_parse_dir = os.path.join(processed_folder, "image-parse")
    if not os.path.exists(processed_image_parse_dir):
        os.makedirs(processed_image_parse_dir)

    processed_image_parse_new_dir = os.path.join(processed_folder, "image-parse-new")
    if not os.path.exists(processed_image_parse_new_dir):
        os.makedirs(processed_image_parse_new_dir)


    processed_image_mask_dir = os.path.join(os.path.join(processed_folder, "image-mask"))
    if not os.path.exists(processed_image_mask_dir):
        os.makedirs(processed_image_mask_dir)

    processed_cloth_mask_dir = os.path.join(os.path.join(processed_folder, "cloth-mask"))
    if not os.path.exists(processed_cloth_mask_dir):
        os.makedirs(processed_cloth_mask_dir)

    processed_pose_dir = os.path.join(os.path.join(processed_folder, "pose"))
    if not os.path.exists(processed_pose_dir):
        os.makedirs(processed_pose_dir)


    
    size = (args.fine_width, args.fine_height)

    from graphonomy.networks import deeplab_xception_transfer
    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20, hidden_layers=128, source_classes=7 )
    net.load_source_model(torch.load(GRAPHONOMY_MODEL_PATH))
    print('load model:', GRAPHONOMY_MODEL_PATH)
    if not args.gpu_ids == '':
        net.cuda()
    else:
        use_gpu = False
        raise RuntimeError('GPU is required for Graphonomy')
    
    
    for img_name,cloth_name in image_names:
        img_path = os.path.join(RAW_FOLDER,"image",img_name)
        cloth_path = os.path.join(RAW_FOLDER,"cloth",cloth_name)

        #read raw images and resize them
        img = read_image(img_path,size = size)
        cloth = read_image(cloth_path,size = size)
        img.save(processed_image_dir+"/"+ img_name)
        cloth.save(processed_cloth_dir+"/"+ cloth_name)

        #use the new path to read
        img_path = os.path.join(processed_image_dir,img_name)
        cloth_path = os.path.join(processed_cloth_dir,cloth_name)

        #create image-parse-new and iamge-pars
        inference(net, img_path, processed_image_parse_dir, processed_image_parse_new_dir, output_name=img_name, use_gpu=True)  

        # #create masks
        make_cloth_mask(cloth_path, os.path.join(processed_cloth_mask_dir,cloth_name), viz=False)
        make_body_mask(processed_image_dir, processed_image_parse_new_dir, img_name, processed_image_mask_dir)
    find_keypoints()
    #create pose with openpose python API 
    

def make_try_on(opt, processed_folder):
    #create dataset for tom
    opt.stage, opt.name ="GMM", "GMM"
    print("\n\nStart to test stage: %s, named: %s!" % (opt.stage, opt.name))
    test_dataset = CPDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)

    #create model gmm
    model_gmm = GMM(opt)
    model_gmm.cuda()
    model_gmm.eval()
    load_checkpoint(model_gmm, opt.checkpoint_GMM)
    
    #create required folders for GMM
    base_name = os.path.basename(opt.checkpoint_GMM)
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)

    with torch.no_grad():
        print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
        for step, inputs in enumerate(test_loader):
            
            iter_start_time = time.time()
        

            c_names = inputs['c_name']
            im_names = inputs['im_name']
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()
            shape_ori = inputs['shape_ori']  # original body shape without blurring

            grid, theta = model_gmm(agnostic, cm)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
            overlay = 0.7 * warped_cloth + 0.3 * im

            visuals = [[im_h, shape, im_pose],
                        [c, warped_cloth, im_c],
                        [warped_grid, (warped_cloth+im)*0.5, im]]

            save_images(warped_cloth, im_names, warp_cloth_dir)
            save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
            save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                        0.8, im_names, result_dir1)
            save_images(warped_grid, im_names, warped_grid_dir)
            save_images(overlay, im_names, overlayed_TPS_dir)

            if (step+1) % opt.display_count == 0:
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f' % (step+1, t), flush=True)

    target = processed_folder.replace("\\","/")
    warp_cloth_folder = os.path.join(opt.result_dir, opt.name, opt.datamode, 'warp-cloth')
    warp_mask_folder = os.path.join(opt.result_dir, opt.name, opt.datamode, 'warp-mask')
    print("moving results folder to %s" % target)
    shutil.rmtree(target+"/warp-cloth")
    shutil.rmtree(target+"/warp-mask")
    shutil.move(warp_cloth_folder, target)
    shutil.move(warp_mask_folder, target)


    
    #create dataset for tom
    opt.stage, opt.name= "TOM","TOM"
    print("\n\n Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    #recreate dataset with folder of warped cloth and mask
    test_dataset = CPDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    model_tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+

    #create model TOM and load checkpoint
    model_tom.cuda()
    model_tom.eval()
    checkpoint_tom= "checkpoints/TOM/tom_final.pth"
    load_checkpoint(model_tom, checkpoint_tom)

    #create required folders for TOM
    base_name = os.path.basename(opt.checkpoint_TOM)
    # save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data

    #make predictions with TOM model
    with torch.no_grad():
        print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
        for step, inputs in enumerate(test_loader):
            iter_start_time = time.time()

            im_names = inputs['im_name']
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']

            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()

            # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
            outputs = model_tom(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = c * m_composite + p_rendered * (1 - m_composite)

            #save images
            save_images(p_tryon, im_names, try_on_dir)
            save_images(im_h, im_names, im_h_dir)
            save_images(shape, im_names, shape_dir)
            save_images(im_pose, im_names, im_pose_dir)
            save_images(m_composite, im_names, m_composite_dir)
            save_images(p_rendered, im_names, p_rendered_dir)  # For test data

            if (step+1) % opt.display_count == 0:
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f' % (step+1, t), flush=True)

    print("Done! Reruslt are available in %s" % save_dir)




def main():
    # parse options and 
    args= get_args()
    label_path = Path(args.dataroot,args.data_list)
    processed_folder= os.path.join(args.dataroot,"processed")


    image_names = []
    for line in label_path.open():
        line=line.strip()
        img_path = line.split(' ')[0]
        cloth_path= line.split(' ')[1]
        image_names.append((img_path,cloth_path))

    if len(image_names)==0:
        print("No images found in %s" % label_path)
        return    

    print("Start to process images..")
    process_images(image_names,args, processed_folder)
    
    make_try_on(args,processed_folder)

    
    


if __name__ == '__main__':
    main()