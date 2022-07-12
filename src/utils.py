import argparse
import os
import cv2 
import numpy as np
from skimage.metrics import structural_similarity
import lpips
from torchvision.transforms import ToTensor, transforms

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="1")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    # parser.add_argument("--datamode", default="train")
    parser.add_argument("--datamode", default="test")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    # parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--data_list", default="test_pairs.txt")
    # parser.add_argument("--data_list", default="test_pairs_same.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str, default='tensortest', help='save tensorboard infos')

    parser.add_argument('--result_dir', type=str,default='result', help='save result infos')

    parser.add_argument('--checkpoint_GMM', type=str, default='checkpoints/GMM/gmm_final.pth', help='model checkpoint for test')
    parser.add_argument('--checkpoint_TOM', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')

    parser.add_argument("--display_count", type=int, default=5)
    parser.add_argument("--shuffle", action='store_true',default=True,
                        help='shuffle input data')

    opt = parser.parse_args(args=[])
    return opt

def move_warp(opt):
    import shutil
    target = os.path.join(opt.dataroot, opt.datamode)

    #deletes folders if they exist
    if os.path.exists(os.path.join(target, "warp-cloth")):
        shutil.rmtree(os.path.join(target, "warp-cloth"))
    if os.path.exists(os.path.join(target, "warp-mask")):
        shutil.rmtree(os.path.join(target, "warp-mask"))

    #moves files
    print("warp-mask moved to:", shutil.move("result/GMM/test/warp-mask", target))
    print("warp-cloth moved to:", shutil.move("result/GMM/test/warp-cloth", target))


def calculate_metrics(opt):
    #initialize transform to normalize between [-1,1]
    transform = transforms.Compose([ToTensor(),transforms.Normalize( mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    target_dir = 'result/TOM/test/try-on'
    initial_dir = 'data/test/image'
    print("calculating SSIM and LPIPS...")

    target_dir = os.path.join(opt.result_dir, "TOM", opt.datamode,"try-on" )
    initial_dir = os.path.join(opt.dataroot,opt.datamode,'image')

    count = 0
    tot_ssim_score = 0
    tot_lpips=0
    for filename in os.listdir(target_dir):
        
        imageA = cv2.imread(os.path.join(target_dir,filename))
        imageB = cv2.imread(os.path.join(initial_dir,filename))
        

        # 4. Convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
        #visualization purpose
        # numpy_horizontal = np.hstack((imageB , imageA))
        # cv2.imshow('Numpy Horizontal', numpy_horizontal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

        # 5. Compute the Structural Similarity Index (SSIM) between the two
        #    images, ensuring that the difference image is returned

        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        
        tot_ssim_score += score
        
        # 6. Compute the LPIPS between the two images
        img0 = transform(imageA) # image should be RGB, IMPORTANT: normalized to [-1,1]
        img1 = transform(imageB)
        d = loss_fn_alex(img0, img1)
        tot_lpips += d

        count += 1

    ssim_score = tot_ssim_score / count
    lpips_score = tot_lpips / count
    lpips_score = lpips_score.item()
    print("SSIM:", ssim_score)
    print("LPIPS: {}".format(lpips_score))

    


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="1")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    # parser.add_argument("--datamode", default="train")
    parser.add_argument("--datamode", default="processed")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    parser.add_argument("--data_list", default="wild_pairs.txt")
    # parser.add_argument("--data_list", default="test_pairs_same.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str, default='tensortest', help='save tensorboard infos')

    parser.add_argument('--result_dir', type=str,default='result', help='save result infos')

    parser.add_argument('--checkpoint_GMM', type=str, default='checkpoints/GMM/gmm_final.pth', help='model checkpoint for test')
    parser.add_argument('--checkpoint_TOM', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')

    parser.add_argument("--display_count", type=int, default=5)
    parser.add_argument("--shuffle", action='store_true',default=True,
                        help='shuffle input data')

    opt = parser.parse_args(args=[])
    return opt