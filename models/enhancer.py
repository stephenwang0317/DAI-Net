import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

# This Retinex Decom Net is frozen during training of DAI-Net
class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet = DecomNet()

    def forward(self, input):
        # Forward DecompNet
        R, I = self.DecomNet(input)
        return R, I

    def loss(self, R_low, I_low, R_high, I_high, input_low, input_high):
        # Compute losses
        recon_loss_low = F.l1_loss(R_low * I_low, input_low)
        recon_loss_high = F.l1_loss(R_high * I_high, input_high)
        recon_loss_mutal_low = F.l1_loss(R_high * I_low, input_low)
        recon_loss_mutal_high = F.l1_loss(R_low * I_high, input_high)
        equal_R_loss = F.l1_loss(R_low, R_high.detach())

        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)

        loss_Decom = recon_loss_low + \
                     recon_loss_high + \
                     0.001 * recon_loss_mutal_low + \
                     0.001 * recon_loss_mutal_high + \
                     0.1 * Ismooth_loss_low + \
                     0.1 * Ismooth_loss_high + \
                     0.01 * equal_R_loss
        return loss_Decom

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def load(self,ckpt_path):
        if os.path.isfile(ckpt_path):
            ckpt_dict =  torch.load(ckpt_path)
            self.DecomNet.load_state_dict(ckpt_dict)
            return True
        else:
            return False

    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_path):
        if self.load(ckpt_path):
            print(ckpt_path, " : loaded")
        else:
            raise Exception
        
        output_arrs =[] 
        for idx in range(len(test_low_data_names)):
            test_img_path  = test_low_data_names[idx]
            test_img_name  = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            
            test_low_img   = Image.open(test_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            input_low  = Variable(torch.FloatTensor(torch.from_numpy(input_low_test))).cuda()
            
            R_low, I_low = self.DecomNet(input_low)
            I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)

            t1 = R_low.detach().cpu()
            t2 = I_low_3.detach().cpu()
            R_low = np.squeeze(t1)
            I_low = np.squeeze(t2)
            R_low= np.concatenate([R_low], axis=2)
            I_low= np.concatenate([I_low], axis=2)
            
            R_low = np.transpose(R_low, (1,2,0))
            I_low = np.transpose(I_low, (1,2,0))

            R_low = Image.fromarray(np.clip(R_low * 255.0, 0, 255.0).astype('uint8'))
            I_low = Image.fromarray(np.clip(I_low * 255.0, 0, 255.0).astype('uint8'))

            filepath = res_dir + '/' + test_img_name
            R_low.save(filepath[:-4] + '_R.jpg')
            I_low.save(filepath[:-4] + '_I.jpg')
            

            output_arrs.append([R_low,I_low])
        
        return output_arrs
        
        


