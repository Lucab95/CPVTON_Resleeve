import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.visualization.visualize import save_images
from src.models.networks import GMM, UnetGenerator

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    #create necessary folders
    base_name = os.path.basename(opt.checkpoint_GMM)
    name = opt.name
    save_dir = os.path.join(opt.result_dir, name, opt.datamode)
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

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                    [c, warped_cloth, im_c],
                    [warped_grid, (warped_cloth+im)*0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)


        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True) 

def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()

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
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2*cm-1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
    