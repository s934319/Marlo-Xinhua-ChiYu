import torch
import torch.nn as nn
import torch.nn.functional as F

# import cv2


class OpticalFlow_s(nn.Module):
    def __init__(self, obs_size, n_frameStack):
        super(OpticalFlow_s, self).__init__()


        # self.hiden_size = hiden_size
        self.obs_size = obs_size
        # self.act_size = act_size

        

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
        # flow_loss is used for training flow network.
        # pred_error is used as the flow-based intrinsic signal.


        # Get the last frame, since optical flow need only two frames.
        # obs -> (batch, framestack*C, 84, 84)
        # obs -> (128, 12, 84, 84)
        # frame -> (batch, C, 84, 84)
        # frame -> (128, 3, 84, 84)
        # N, C, H, W
        # print(obs.shape)
        # print(next_obs.shape)
        frame = obs[:, -self.channel:, :, :]
        next_frame = next_obs[:, -self.channel:, :, :]

        h = obs.shape[2]
        w = obs.shape[3]


        # Divide 255.0, let the input observation range in [0, 1] (Take as warping input)
        frame = frame/255.0
        next_frame = next_frame/255.0


        # print(frame.shape)
        # print(next_frame.shape)


        # Input for neural network input with mean-zero values in [-1, 1] (Take as network input)
        frame_normalized = (frame - self.obs_mean[-self.channel:, :, :]) / self.obs_std
        next_frame_normalized = (next_frame - self.obs_mean[-self.channel:, :, :]) / self.obs_std

        obs_stack_fw = torch.cat( (frame_normalized, next_frame_normalized), 1)
        obs_stack_bw = torch.cat( (next_frame_normalized, frame_normalized), 1)
        # print(obs_stack_fw.shape)

        # features_fw = self.get_flowS_features(obs_stack_fw, fix_features)
        # flow_fw = self.flowS(features_fw[0], features_fw[1], features_fw[2])
        flow_fw = self.forward(obs_stack_fw)

        # features_bw = self.get_flowS_features(obs_stack_bw, fix_features)
        # flow_bw = self.flowS(features_bw[0], features_bw[1], features_bw[2])
        flow_bw = self.forward(obs_stack_bw)

        # print(flow_fw.shape)

        ## Optical flow for training flow module
        # self.flow_fw_up = tf.image.resize_bilinear(flow_fw, [self.h, self.w]) * 5.0
        # self.flow_bw_up = tf.image.resize_bilinear(flow_bw, [self.h, self.w]) * 5.0
        # flow_fw_up = cv2.resize(flow_fw, (w, h)) * 5.0
        # flow_bw_up = cv2.resize(flow_bw, (w, h)) * 5.0
        flow_fw_up = F.interpolate(flow_fw, size=(w, h), mode='bilinear') * 5.0
        flow_bw_up = F.interpolate(flow_bw, size=(w, h), mode='bilinear') * 5.0

        # print(next_frame.shape)
        # print(flow_fw_up.shape)
        _frame = image_warp(next_frame, flow_fw_up, device)
        _next_frame = image_warp(frame, flow_bw_up, device)
        # print(_frame.shape)


        # fw_diff_ob = tf.reshape((frame - _frame), self.obs_sh) * self.beta
        # bw_diff_ob = tf.reshape((next_frame - _next_frame), self.obs_sh) * self.beta
        fw_diff_frame = (frame - _frame) * self.beta
        bw_diff_frame = (next_frame - _next_frame) * self.beta
        # print(fw_diff_frame.shape)

        # fw_loss_frame = tf.pow(tf.square(fw_diff_frame) + tf.square(self.epsilon), self.alpha)
        # bw_loss_frame = tf.pow(tf.square(bw_diff_frame) + tf.square(self.epsilon), self.alpha)
        fw_loss_frame = torch.pow(torch.square(fw_diff_frame) + self.epsilon**2, self.alpha)
        bw_loss_frame = torch.pow(torch.square(bw_diff_frame) + self.epsilon**2, self.alpha)
        # print(fw_loss_frame.shape)

        # pred_error = tf.reduce_mean(fw_loss_frame, axis=[2, 3, 4]) + tf.reduce_mean(bw_loss_frame, axis=[2, 3, 4])
        # flow_loss = tf.reduce_mean(fw_loss_frame + bw_loss_frame)
        pred_error = torch.mean(fw_loss_frame, dim=[1, 2, 3]) + torch.mean(bw_loss_frame, dim=[1, 2, 3])
        # print(pred_error.shape)
        flow_loss = torch.mean(fw_loss_frame + bw_loss_frame)
        # print(flow_loss.shape)


        # flow_loss is used for training flow network.
        # pred_error is used as the flow-based intrinsic signal.
        return flow_loss, pred_error



    def forward(self, x):

        l1_x = F.elu(self.l1(x))
        l2_x = F.elu(self.l2(l1_x))
        l3_x = F.elu(self.l3(l2_x))
        l4_x = F.elu(self.l4(l3_x))


        # dl2_x = tf.nn.elu(my_deconv2d(l4_x, 64, [3, 3], stride=2, out_shape=[11, 11], c_i=96, name='dl2'))
        # concat2 = tf.concat([l3_x, dl2_x], axis=3)
        dl2_x = F.elu(self.deconv1(l4_x))
        concat2 = torch.cat( (l3_x, dl2_x), 1)


        # dl1_x = tf.nn.elu(my_deconv2d(concat2, 32, [3, 3], stride=2, out_shape=[21, 21], c_i=128, name='dl1'))
        # concat1 = tf.concat([l2_x, dl1_x], axis=3)
        dl1_x = F.elu(self.deconv2(concat2))
        concat1 = torch.cat( (l2_x, dl1_x), 1)


        # flow = slim.conv2d(concat1, 2, [3, 3], activation_fn=None, stride=1)
        flow = self.conv4(concat1)

        return flow

    def set(self, obs_mean, obs_std):
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        # Since Optical flow input range in [0, 1]
        self.obs_mean = self.obs_mean / 255.0 
        self.obs_std = self.obs_std / 255.0



def image_warp(im, flow, device):
    """Performs a backward warp of an image using the predicted flow.
    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # num_batch, height, width, channels = tf.unstack(tf.shape(im))
    # num_batch, channels, height, width = im.shape
    im = im.permute(0, 2, 3, 1)
    num_batch, height, width, channels = im.shape

    # max_x = tf.cast(width - 1, 'int32')
    # max_y = tf.cast(height - 1, 'int32')
    # zero = tf.zeros([], dtype='int32')
    max_x = width - 1
    max_y = height - 1

    # We have to flatten our tensors to vectorize the interpolation
    # im_flat = tf.reshape(im, [-1, channels])
    # flow_flat = tf.reshape(flow, [-1, 2])
    im_flat = torch.reshape(im, (-1, channels))
    flow_flat = torch.reshape(flow, (-1, 2))

    # Floor the flow, as the final indices are integers
    # The fractional part is used to control the bilinear interpolation.
    # flow_floor = tf.to_int32(tf.floor(flow_flat))
    # bilinear_weights = flow_flat - tf.floor(flow_flat)
    flow_floor = torch.floor(flow_flat).int()
    bilinear_weights = flow_flat - torch.floor(flow_flat)

    # Construct base indices which are displaced with the flow
    # pos_x = tf.tile(tf.range(width), [height * num_batch])
    # grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
    # pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])
    pos_x = torch.arange(width, device=device).repeat(height * num_batch)
    grid_y = torch.unsqueeze(torch.arange(height, device=device), 1).repeat(1, width)
    pos_y = torch.reshape(grid_y, [-1]).repeat(num_batch)

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]

    # Compute interpolation weights for 4 adjacent pixels
    # expand to num_batch * height * width x 1 for broadcasting in add_n below
    # wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
    # wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
    # wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
    # wd = tf.expand_dims(xw * yw, 1) # bottom right pixel
    wa = torch.unsqueeze((1 - xw) * (1 - yw), 1) # top left pixel
    wb = torch.unsqueeze((1 - xw) * yw, 1) # bottom left pixel
    wc = torch.unsqueeze(xw * (1 - yw), 1) # top right pixel
    wd = torch.unsqueeze(xw * yw, 1) # bottom right pixel

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1

    # x0 = tf.clip_by_value(x0, zero, max_x)
    # x1 = tf.clip_by_value(x1, zero, max_x)
    # y0 = tf.clip_by_value(y0, zero, max_y)
    # y1 = tf.clip_by_value(y1, zero, max_y)
    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)

    dim1 = width * height
    # batch_offsets = tf.range(num_batch) * dim1
    # base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
    # base = tf.reshape(base_grid, [-1])
    batch_offsets = torch.arange(num_batch, device=device) * dim1
    base_grid = torch.unsqueeze(batch_offsets, 1).repeat(1, dim1)
    base = torch.reshape(base_grid, [-1])

    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    # idx_a = base_y0 + x0
    # idx_b = base_y1 + x0
    # idx_c = base_y0 + x1
    # idx_d = base_y1 + x1
    idx_a = (base_y0 + x0).repeat(3, 1).permute(1, 0)
    idx_b = (base_y1 + x0).repeat(3, 1).permute(1, 0)
    idx_c = (base_y0 + x1).repeat(3, 1).permute(1, 0)
    idx_d = (base_y1 + x1).repeat(3, 1).permute(1, 0)


    # Ia = tf.gather(im_flat, idx_a)
    # Ib = tf.gather(im_flat, idx_b)
    # Ic = tf.gather(im_flat, idx_c)
    # Id = tf.gather(im_flat, idx_d)
    Ia = torch.gather(im_flat, 0, idx_a)
    Ib = torch.gather(im_flat, 0, idx_b)
    Ic = torch.gather(im_flat, 0, idx_c)
    Id = torch.gather(im_flat, 0, idx_d)

    # warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    # warped = tf.reshape(warped_flat, [num_batch, height, width, channels])
    warped_flat = wa*Ia + wb*Ib + wc*Ic + wd*Id
    warped = torch.reshape(warped_flat, [num_batch, height, width, channels])


    # num_batch, height, width, channels
    # -> num_batch, channels, height, width
    warped = warped.permute(0, 3, 1, 2)

    return warped