import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlow_s(nn.Module):
    def __init__(self, obs_size, n_frameStack):
        super(OpticalFlow_s, self).__init__()

        self.obs_size = obs_size

        self.channel = int(self.obs_size[0]/n_frameStack)


        self.alpha = 0.45
        self.beta = 255
        self.epsilon = 0.001


        self.l1 = nn.Conv2d(self.channel*2, 32, 4, stride=2)
        self.l2 = nn.Conv2d(32, 64, 4, stride=2)
        self.l3 = nn.Conv2d(64, 64, 3, stride=2)
        self.l4 = nn.Conv2d(64, 96, 3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(96, 64, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64+64, 32, 3, stride=2)

        self.conv4 = nn.Conv2d(64+32, 2, 3, stride=1)



    def compute_loss(self, obs, next_obs, device):
        # obs -> (batch, framestack*C, 84, 84)
        # obs -> (128, 12, 84, 84)
        # N, C, H, W
        frame = obs[:, -self.channel:, :, :]
        next_frame = next_obs[:, -self.channel:, :, :]

        h = obs.shape[2]
        w = obs.shape[3]


        # Divide 255.0, let the input observation range in [0, 1] (Take as warping input)
        frame = frame/255.0
        next_frame = next_frame/255.0


        frame_normalized = frame
        next_frame_normalized = next_frame

        obs_stack_fw = torch.cat( (frame_normalized, next_frame_normalized), 1)
        obs_stack_bw = torch.cat( (next_frame_normalized, frame_normalized), 1)


        flow_fw = self.forward(obs_stack_fw)
        flow_bw = self.forward(obs_stack_bw)


        ## Optical flow for training flow module
        flow_fw_up = F.interpolate(flow_fw, size=(w, h), mode='bilinear') * 5.0
        flow_bw_up = F.interpolate(flow_bw, size=(w, h), mode='bilinear') * 5.0


        _frame = image_warp(next_frame, flow_fw_up, device)
        _next_frame = image_warp(frame, flow_bw_up, device)


        fw_diff_frame = (frame - _frame) * self.beta
        bw_diff_frame = (next_frame - _next_frame) * self.beta


        
        fw_loss_frame = torch.pow(torch.square(fw_diff_frame) + self.epsilon**2, self.alpha)
        bw_loss_frame = torch.pow(torch.square(bw_diff_frame) + self.epsilon**2, self.alpha)


        # flow_loss is used for training flow network.
        # pred_error is used as the flow-based intrinsic signal.
        pred_error = torch.mean(fw_loss_frame, dim=[1, 2, 3]) + torch.mean(bw_loss_frame, dim=[1, 2, 3])
        flow_loss = torch.mean(fw_loss_frame + bw_loss_frame)
        
        return flow_loss, pred_error



    def forward(self, x):
        l1_x = F.elu(self.l1(x))
        l2_x = F.elu(self.l2(l1_x))
        l3_x = F.elu(self.l3(l2_x))
        l4_x = F.elu(self.l4(l3_x))


        dl2_x = F.elu(self.deconv1(l4_x))
        concat2 = torch.cat( (l3_x, dl2_x), 1)


        dl1_x = F.elu(self.deconv2(concat2))
        concat1 = torch.cat( (l2_x, dl1_x), 1)


        flow = self.conv4(concat1)

        return flow


def image_warp(im, flow, device):
    """Performs a backward warp of an image using the predicted flow.
    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    im = im.permute(0, 2, 3, 1)
    num_batch, height, width, channels = im.shape


    max_x = width - 1
    max_y = height - 1

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = torch.reshape(im, (-1, channels))
    flow_flat = torch.reshape(flow, (-1, 2))

    # Floor the flow, as the final indices are integers
    # The fractional part is used to control the bilinear interpolation.
    flow_floor = torch.floor(flow_flat).int()
    bilinear_weights = flow_flat - torch.floor(flow_flat)

    # Construct base indices which are displaced with the flow
    pos_x = torch.arange(width, device=device).repeat(height * num_batch)
    grid_y = torch.unsqueeze(torch.arange(height, device=device), 1).repeat(1, width)
    pos_y = torch.reshape(grid_y, [-1]).repeat(num_batch)

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]

    # Compute interpolation weights for 4 adjacent pixels
    # expand to num_batch * height * width x 1 for broadcasting in add_n below
    wa = torch.unsqueeze((1 - xw) * (1 - yw), 1) # top left pixel
    wb = torch.unsqueeze((1 - xw) * yw, 1) # bottom left pixel
    wc = torch.unsqueeze(xw * (1 - yw), 1) # top right pixel
    wd = torch.unsqueeze(xw * yw, 1) # bottom right pixel

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1


    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)

    dim1 = width * height

    batch_offsets = torch.arange(num_batch, device=device) * dim1
    base_grid = torch.unsqueeze(batch_offsets, 1).repeat(1, dim1)
    base = torch.reshape(base_grid, [-1])

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width

    assert channels==1 or channels==3
    if(channels==1):
        idx_a = (base_y0 + x0).view(-1, 1)
        idx_b = (base_y1 + x0).view(-1, 1)
        idx_c = (base_y0 + x1).view(-1, 1)
        idx_d = (base_y1 + x1).view(-1, 1)
    else:
        idx_a = (base_y0 + x0).repeat(3, 1).permute(1, 0)
        idx_b = (base_y1 + x0).repeat(3, 1).permute(1, 0)
        idx_c = (base_y0 + x1).repeat(3, 1).permute(1, 0)
        idx_d = (base_y1 + x1).repeat(3, 1).permute(1, 0)


    Ia = torch.gather(im_flat, 0, idx_a)
    Ib = torch.gather(im_flat, 0, idx_b)
    Ic = torch.gather(im_flat, 0, idx_c)
    Id = torch.gather(im_flat, 0, idx_d)


    warped_flat = wa*Ia + wb*Ib + wc*Ic + wd*Id
    warped = torch.reshape(warped_flat, [num_batch, height, width, channels])


    # num_batch, height, width, channels
    # -> num_batch, channels, height, width
    warped = warped.permute(0, 3, 1, 2)

    return warped