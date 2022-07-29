import random
import os
import os.path as osp
import sys
sys.path.append('./')
import math
import copy 
import pprint
import shutil
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
import copy
import pandas as pd
import seaborn as sns
import imgaug.augmenters as iaa
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet152, densenet121, mobilenet_v2


# from DiffableMaps import ExpModel
from integrated_gradients import IntegratedGradients
from xrai import XRAI

def load_model(model_info, cfgs):
    '''model + optimer initialization'''
    model_name = model_info['model_name']
    model_path = model_info['model_path']
    
    model_cfgs = cfgs['model']
    data_cfgs = cfgs['dataset']
            
    if model_name=='densenet121':
        model = densenet121(pretrained=model_cfgs['pretrained'])
        ### 3-channel --> 1-channel
        old_module = model.features.conv0
        model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=old_module.out_channels, 
                                         kernel_size=old_module.kernel_size, stride=old_module.stride, 
                                         padding=old_module.padding, dilation=old_module.dilation, 
                                         bias=old_module.bias)
        ### IN output (1000) --> CX output (14)
        model.classifier = nn.Linear(model.classifier.in_features, out_features=data_cfgs['n_classes'])
        nn.init.constant_(model.classifier.bias, 0)
        model = model.to(torch.device(cfgs['device']))
        model = torch.nn.DataParallel(model)

    elif model_name=='mobilenet_v2':
        model = mobilenet_v2(pretrained=model_cfgs['pretrained'])
        ### 3-channel --> 1-channel
        old_module = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=old_module.out_channels, 
                                        kernel_size=old_module.kernel_size, stride=old_module.stride, 
                                        padding=old_module.padding, dilation=old_module.dilation, 
                                        bias=old_module.bias)
        ### IN output (1000) --> CX output (14)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features=data_cfgs['n_classes'])
        model = torch.nn.DataParallel(model)

    elif model_name=='resnet152':
        model = resnet152(pretrained=model_cfgs['pretrained'])
        ### 3-channel --> 1-channel
        old_module = model.conv1
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=old_module.out_channels, 
                                kernel_size=old_module.kernel_size, stride=old_module.stride, 
                                padding=old_module.padding, dilation=old_module.dilation, 
                                bias=old_module.bias)
        ### IN output (1000) --> CX output (14)
        model.fc = nn.Linear(model.fc.in_features, out_features=data_cfgs['n_classes'])
        model = torch.nn.DataParallel(model)

    else:
        raise RuntimeError('Model architecture not supported.')
    
#     model = model.to(torch.device(cfgs['device']))

#     print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
#             model._get_name(), sum(p.numel() for p in model.parameters()), cfgs['step']))
    
#     print('Restoring model weights from {}'.format(model_path))
    model_checkpoint = torch.load(model_path, map_location=cfgs['device'])
    model.load_state_dict(model_checkpoint['state_dict'])
    cfgs['step'] = model_checkpoint['global_step']
    del model_checkpoint
#     model = model.module
    return model


class ExpModel(ABC):
    '''the base class containing all XAI methods'''
    
    Exp_Methods = ['VanillaBP', 'VanillaBP_Img', 'GuidedBP', 'IntegratedBP', 'GradCAM', 'GuidedGradCam', 'SmoothBP', 'XRAI']
    def __init__(self, model_info=None, cfgs=None, init_from=None):
        assert (model_info is not None) or (init_from is not None)
        if model_info is not None:
            assert cfgs is not None
            self.model = load_model(model_info, cfgs)
            self.model.eval()
            self.model_name = model_info['model_name']
        else:
            self.model = init_from.model
            self.model_name = init_from.model_name
        self.device = cfgs['device']
        
        self.hook_handles = []
        self.hooked = False
        self.actF_current = None
        self.actF_args = None
        self.actF_changed = False
        
        
    '''convert activation function'''   
    def convert_act(self, target_acts, new_act, **kwargs): ### inner control
        target_acts = tuple(target_acts)
        self.actF_changed = True
        self.actF_current = new_act
        self.actF_args = kwargs
        queue = [self.model]
        while len(queue) > 0:
            moi = queue.pop()
            for _m_name, _m in moi.named_children():
                if len(list(_m.children())) > 0:
                    queue.append(_m)
                elif isinstance(_m, target_acts):
                    setattr(moi, _m_name, new_act(**kwargs))
    
    
    '''FP --> logits |||| FP --> max index, max prob'''
    def forward(self, input_image):
#         print(type(self.model))
#         print(input_image.size())
        output = self.model(input_image)
        return output
    
    
    '''Remove all the forward/backward hook functions'''
    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hooked = False
    
    @abstractmethod
    def cal_exp_map(self, input_image, class_of_interest):
        pass
    
class VanillaBPModel(ExpModel):
    
    def __init__(self, model_info, cfgs):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.exp_name = 'VanillaBP'
        
        
    def cal_exp_map(self, input_image, class_of_interest, norm=True):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self.model.zero_grad()
        
        output= self.forward(input_image)
        loss = class_of_interest.mul(output).sum()
        
        heatmap = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        assert len(heatmap.size()) == 4
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        if norm:
            heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        return heatmap, output
    
    
class VanillaBP_ImgModel(ExpModel):
    
    def __init__(self, model_info, cfgs):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.exp_name = ' VanillaBP_Img'
        
        
    def cal_exp_map(self, input_image, class_of_interest):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self.model.zero_grad()
        
        output= self.forward(input_image)
        loss = class_of_interest.mul(output).sum()
        
        heatmap = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        assert len(heatmap.size()) == 4
        heatmap = heatmap * input_image
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        return heatmap, output
    
    
class GradCAMModel(ExpModel):
    def __init__(self, model_info=None, cfgs=None, init_from=None, target_layer=None):
#     def __init__(self, cfgs, target_layer, init_from=None, model_info=None):
        super().__init__(model_info=model_info, cfgs=cfgs, init_from=init_from)
        self.exp_name = 'GradCAM'
        self.vis_fmap = None
        
        if not isinstance(target_layer, list):
            target_layer = [target_layer]
        self.target_layer = target_layer
        
    
    '''Hook activation functions'''    
    def __forward_hook(self, module, tensor_in, tensor_out): # Store results of forward pass
        self.vis_fmap = tensor_out
    
    
    '''GradCam: hook fp only'''
    def __register_hook(self):
        _model = self.model.module
        for i, _loi in enumerate(self.target_layer):
#             print('tofind', _loi)
#             print('data type', type(_model))
            for _name, _module in _model._modules.items():
                if _name == _loi:
                    if i == len(self.target_layer) - 1:
                        self.hook_handles.append(_module.register_forward_hook(self.__forward_hook))
                        self.hooked = True
                    _model = _module
#                     print('finded', _loi, _model)
                    break
        return self.hooked

    
    def cal_exp_map(self, input_image, class_of_interest, norm=True):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self.vis_fmap = None
        self.remove_hook()
        self.__register_hook()
        self.model.zero_grad()
        
        output = self.forward(input_image)
        loss = class_of_interest.mul(output).sum()
        
#         print(self.vis_fmap)
#         print(loss)
        grads = torch.autograd.grad(loss, [self.vis_fmap], retain_graph=True, create_graph=True)[0]
        weights = F.adaptive_avg_pool2d(grads, 1)
        heatmap = torch.mul(self.vis_fmap, weights).sum(dim=1, keepdim=True)
        if self.actF_current is not None:
            heatmap = self.actF_current(**self.actF_args)(heatmap)
        else:
            heatmap = F.relu(heatmap)
#         print(heatmap.size())
        
        heatmap = F.interpolate(heatmap, input_image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = torch.sum(torch.abs(heatmap), dim=1)# dim reduction
#         if _log:
#             print(heatmap.min(), heatmap.max(), heatmap.sum())
        if norm:
            heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        self.vis_fmap = None
        return heatmap, output
    

class GuidedBPModel(ExpModel):
    
    def __init__(self, model_info=None, cfgs=None, init_from=None):
        super().__init__(model_info=model_info, cfgs=cfgs, init_from=init_from)
        self.exp_name = 'GuidedBP'
        self._fmaps = []
        
    
    '''Hook activation functions'''    
    def __forward_hook(self, module, tensor_in, tensor_out): # Store results of forward pass
        self._fmaps.append(tensor_out)
    
    
    def __backward_hook(self, module, grad_in, grad_out):
        if isinstance(module, nn.Softplus):
            cfmap = self._fmaps.pop() # module.ten_out
            beta = self.actF_args['beta']
            modified_grad_out = F.softplus(grad_out[0], beta) * torch.sigmoid(beta * cfmap)
        else:
            cfmap = self._fmaps.pop() # module.ten_out
            modified_grad_out =  F.relu(grad_out[0]) * (cfmap > 0)
        return (modified_grad_out,)
    
                    
    '''GradCam: hook fp only'''
    def __register_hook(self):
        assert (self.actF_current is None) or isinstance(self.actF_current, nn.Softplus)
        queue = [self.model.module]
        while len(queue) > 0:
            moi = queue.pop()
            for _m_name, _m in moi.named_children():
                if len(list(_m.children())) > 0:
                    queue.append(_m)
                elif isinstance(_m, nn.Softplus) or isinstance(_m, nn.ReLU):
                    self.hook_handles.append(_m.register_forward_hook(self.__forward_hook)) # FP
                    self.hook_handles.append(_m.register_backward_hook(self.__backward_hook)) # BP
                    self.hooked = True
        return self.hooked
                
    
    def cal_exp_map(self, input_image, class_of_interest):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self._fmaps = []
        self.remove_hook()
        self.__register_hook()
        self.model.zero_grad()
        
        output= self.forward(input_image)
        loss = class_of_interest.mul(output).sum()
        
        ### doulbe size your fmaps for twice bp
#         self._fmaps = self._fmaps + self._fmaps
        
        heatmap = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        assert len(heatmap.size()) == 4
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        self.fmaps_ = []
        return heatmap, output
    
    
class GuidedGradCAMModel(ExpModel):
    
    def __init__(self, model_info, cfgs, target_layer):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.__gradcam = GradCAMModel(cfgs=cfgs, target_layer=target_layer, init_from=self)
        self.__guidedbp = GuidedBPModel(cfgs=cfgs, init_from=self)
        self.exp_name = 'GuidedGradCAM'
        
    def cal_exp_map(self, input_image, class_of_interest):
        hm_gradcam, output = self.__gradcam.cal_exp_map(input_image, class_of_interest)
        hm_guidedbp, _ = self.__guidedbp.cal_exp_map(input_image, class_of_interest)
        
        print('gradcam: ', hm_gradcam.shape)
        print('hm_guidedbp: ', len(hm_guidedbp.shape))
        
        heatmap = hm_gradcam * hm_guidedbp
        print('product: ', len(heatmap.shape))
        
#         assert len(heatmap.size()) == 4
#         heatmap = torch.sum(torch.abs(heatmap), dim=1)
        heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        return heatmap, output
    
    
class IntegratedBPModel(ExpModel):
    
    def __init__(self, model_info, cfgs):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.exp_name = 'IntegratedBP'
        
    def cal_exp_map(self, input_image, class_of_interest, num_summands = 10):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self.model.zero_grad()
        
#         parallel_model = torch.nn.DataParallel(self.model)               
        prefactors = input_image.new_tensor([k / num_summands for k in range(1, num_summands + 1)]) # prefactor 0 to 1
        image_batch = prefactors.view(num_summands, 1, 1, 1) * input_image
        
        output = self.forward(image_batch) ### from (000) linearly incrs to x
        
        ### sum the result and then take the derivative (instead of summing derivatives as in most implementations)
        loss = (1 / num_summands) * class_of_interest.mul(output).sum()
        heatmap = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        assert len(heatmap.size()) == 4
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        return heatmap, output[-1].unsqueeze_(0)
    
    
class SmoothBPModel(ExpModel):
    
    def __init__(self, model_info, cfgs):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.exp_name = 'SmoothBP'
        
    def cal_exp_map(self, input_image, class_of_interest, num_summands = 10, stdev = 0.03):
        input_image.requires_grad=True
        class_of_interest = torch.tensor(class_of_interest).to(torch.device(self.device))
        self.model.zero_grad()
        
#         parallel_model = torch.nn.DataParallel(self.model)
        prefactors = input_image.new_tensor([1 for k in range(1, num_summands + 1)]) # prefactor 0 to 1
        image_batch = prefactors.view(num_summands, 1, 1, 1) * input_image
        noise = torch.Tensor(np.random.normal(0, stdev, image_batch.shape).astype(np.float32)).cuda()
        image_batch += noise
        output = self.forward(image_batch)
        
        ### sum the result and then take the derivative (instead of summing derivatives as in most implementations)
        loss = (1 / num_summands) * class_of_interest.mul(output).sum()
        heatmap = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        assert len(heatmap.size()) == 4
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        heatmap = heatmap / torch.sum(heatmap, dim=(1,2))
        return heatmap, output[-1].unsqueeze_(0)
    
    
class XRAIModel(ExpModel):
    
    def __init__(self, model_info, cfgs, segments=10):
        super().__init__(model_info=model_info, cfgs=cfgs)
        self.exp_name = 'XRAI'
        self.segments = segments

    def cal_exp_map(self, input_image, class_of_interest):
#         input_image.requires_grad=True
        img_out= input_image.cpu().numpy()[0]
        img_out = img_out.transpose(1,2,0)
        img_out = img_out.astype(np.float32)
        xrai_object = XRAI()
        class_of_interest = torch.tensor(class_of_interest).cuda()
        output = self.forward(input_image) #####
        heatmap = xrai_object.GetMask(self.model, 
                                      input_image.detach().cpu().numpy()[0].transpose(1,2,0),
                                      class_of_interest, self.segments
                                      )# Compute XRAI attributions
        return heatmap, output[-1].unsqueeze_(0) #None
    
    
def expmodel_factory(model_cfgs, cfgs):
#     model_bank = {exp_name:getattr(sys.modules[__name__], exp_name+'Model') for exp_name in ExpModel.Exp_Methods}
    exp_method = model_cfgs['exp_method']
    model_info = model_cfgs['model_info']
    exp_cfgs = model_cfgs['exp_cfgs']
#     return model_bank[exp_method](model_info, cfgs, **exp_cfgs)
    return getattr(sys.modules[__name__], exp_method+'Model')(model_info, cfgs, **exp_cfgs)
