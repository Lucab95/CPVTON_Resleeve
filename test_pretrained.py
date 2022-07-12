import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from src.utils import get_opt, move_warp, calculate_metrics
from src.data.cp_dataset import CPDataset, CPDataLoader
from src.models.networks import *
from src.models.predict_model import test_gmm,test_tom
import warnings
warnings.filterwarnings("ignore")


def main():
    opt = get_opt()
    print("params",opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    

    # create dataset
    test_dataset = CPDataset(opt)

    # create dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)

    # create model & test
    model_gmm = GMM(opt)
    load_checkpoint(model_gmm, opt.checkpoint_GMM)
    with torch.no_grad():
        test_gmm(opt, test_loader, model_gmm, "board")


    opt.name, opt.stage = "TOM" , "TOM"

    #move the warp folders to the test folder for TOM
    move_warp(opt)

    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
     # create dataset for Tom, the same as the test dataset but with the warp folder
    test_dataset = CPDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)

    #makes the 
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
    model_tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
    load_checkpoint(model_tom, opt.checkpoint_TOM)
    with torch.no_grad():
        test_tom(opt, test_loader, model_tom, "board")

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

    calculate_metrics(opt)



if __name__ == "__main__":
    main()
