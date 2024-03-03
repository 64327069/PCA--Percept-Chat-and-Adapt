# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch

# def epoch_saving(epoch, model, fusion_model, optimizer, filename):
#     torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'fusion_model_state_dict': fusion_model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model_image ,optimizer, fusion_model= None, fg_model= None):
    best_name = '{}/model_best.pt'.format(working_dir)
    if fusion_model is None :
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_image.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_name)  # just change to your
    
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_image.state_dict(),
            'fusion_model_state_dict': fusion_model.state_dict(),
            'fg_state_dict': fg_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_name)  # just change to your preferred folder/filename