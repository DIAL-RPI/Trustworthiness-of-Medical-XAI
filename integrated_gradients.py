from base import CoreSaliency
from base import INPUT_OUTPUT_GRADIENTS
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class IntegratedGradients(CoreSaliency):
    expected_keys = [INPUT_OUTPUT_GRADIENTS]
    def forward(self, input_image):
        output = self.model(input_image)
        return output
    
    # def classify(self, input_image):
    #     outputs = self.forward(input_image)
    #     return outputs, F.softmax(outputs, dim=1), torch.max(outputs, 1)[1] ### return the (pred prob, pred class)
    
    def GetMask(self, model, input_image, one_hot, num_summands=10):
        input_image = input_image.transpose(2,0,1)
        input_image = torch.Tensor(input_image).unsqueeze(0).cuda()
        
        input_image = input_image.cuda()
        input_image.requires_grad=True
        self.model = model.cuda()
        
        prefactors = input_image.new_tensor([k / num_summands for k in range(1, num_summands + 1)]) # prefactor 0 to 1
        model.zero_grad()
        parallel_model = torch.nn.DataParallel(self.model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * input_image) ### from (000) linearly incrs to x
        
        ### sum the result and then take the derivative (instead of summing derivatives as in most implementations)
        loss = (1 / num_summands) * one_hot.mul(y).sum()
        self.gradients = torch.autograd.grad(loss, [input_image], retain_graph=True, create_graph=True)[0]
        # self.gradients = torch.sum(torch.abs(self.gradients), dim=1)
        # self.gradients = self.gradients / torch.sum(self.gradients)
        x_arr = self.gradients.data.cpu().numpy()[0]
        x_arr = x_arr.transpose(1,2,0)
        return x_arr
    
    def GetMask_org(self, x_value, call_model_function_, call_model_args=None,
              x_baseline=None, x_steps=25, batch_size=1):
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)
            
        assert x_baseline.shape == x_value.shape
        x_diff = x_value - x_baseline
        total_gradients = np.zeros_like(x_value, dtype=np.float32)
        x_step_batched = []
        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff
            x_step_batched.append(x_step)
            if len(x_step_batched) == batch_size or alpha == 1:
                x_step_batched = np.asarray(x_step_batched)
                call_model_output = call_model_function_(
                                    x_step_batched,
                                    call_model_args=call_model_args,
                                    expected_keys=self.expected_keys)
                self.format_and_check_call_model_output(call_model_output,
                                                    x_step_batched.shape,
                                                    self.expected_keys)
                total_gradients += call_model_output[INPUT_OUTPUT_GRADIENTS].sum(axis=0)
                x_step_batched = []
        return total_gradients * x_diff / x_steps