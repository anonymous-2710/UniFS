import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModelWrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super(BaseModelWrapper, self).__init__()
        self.model = model
        self.loss = loss_fn

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        # k_space_cropped = batch_data['k_space_cropped']
        # mask = batch_data['mask']
        t2_sp = self.model(t1_hr, t2_lr)
        if return_preds:
            return t2_sp, t2_hr

        loss = self.loss(t2_sp, t2_hr)
        return loss


class MaskModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(MaskModelWrapper, self).__init__(model, loss_fn)

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        # k_space_cropped = batch_data['k_space_cropped']
        mask = batch_data['mask']
        t2_sp = self.model(t1_hr, t2_lr, mask=mask)
        if return_preds:
            return t2_sp, t2_hr

        loss = self.loss(t2_sp, t2_hr)
        return loss


class KSpaceModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(KSpaceModelWrapper, self).__init__(model, loss_fn)

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        k_space_cropped = batch_data['k_space_cropped']
        # mask = batch_data['mask']
        t2_sp = self.model(t1_hr, t2_lr, k_space_cropped=k_space_cropped)
        if return_preds:
            return t2_sp, t2_hr

        loss = self.loss(t2_sp, t2_hr)
        return loss


class PIPModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(PIPModelWrapper, self).__init__(model, loss_fn)

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        k_space_cropped = batch_data['k_space_cropped']
        # mask = batch_data['mask']
        degradation_class = batch_data['degradation_class']

        t2_sp = self.model(t1_hr, t2_lr, degradation_class=degradation_class)
        if return_preds:
            return t2_sp, t2_hr

        loss = self.loss(t2_sp, t2_hr)
        return loss


class FSMNetModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(FSMNetModelWrapper, self).__init__(model, loss_fn)
        self.loss_function_fre = torch.nn.L1Loss()

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        t2_sp = self.model(t1_hr, t2_lr)
        spatial, fre = t2_sp['img_out'], t2_sp['img_fre']
        if return_preds:
            return spatial, t2_hr

        loss = self.loss(spatial, t2_hr) + self.loss_function_fre(fre, t2_hr)
        return loss


class MINetModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(MINetModelWrapper, self).__init__(model, loss_fn)
        self.loss_function = torch.nn.L1Loss()

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        t1_sp, t2_sp = self.model(t1_hr, t2_lr)
        if return_preds:
            return t2_sp, t2_hr

        loss = 0.3 * self.loss_function(t2_sp, t2_hr) + 0.7 * self.loss_function(t1_sp, t1_hr)
        return loss


class MTransModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(MTransModelWrapper, self).__init__(model, loss_fn)
        self.l1_loss = torch.nn.L1Loss()
        self.cl1_loss = torch.nn.L1Loss()
        self.input_size = 224

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        t1_hr = F.interpolate(t1_hr, (self.input_size, self.input_size), mode="bilinear")
        t2_hr = F.interpolate(t2_hr, (self.input_size, self.input_size), mode="bilinear")
        t2_lr = F.interpolate(t2_lr, (self.input_size, self.input_size), mode="bilinear")

        t2_sp, t1_sp = self.model(t2_lr, t1_hr)
        if return_preds:
            return t2_sp, t2_hr

        l1_loss = self.l1_loss(t2_sp, t2_hr)
        cl1_loss = self.cl1_loss(t1_sp, t1_hr)
        loss = l1_loss + cl1_loss
        return loss


class MCVarNetModelWrapper(BaseModelWrapper):
    def __init__(self, model, loss_fn):
        super(MCVarNetModelWrapper, self).__init__(model, loss_fn)
        self.loss_function = torch.nn.L1Loss()

    def forward(self, batch_data, return_preds=False):
        t1_hr = batch_data['t1_hr']
        t2_hr = batch_data['t2_hr']
        t2_lr = batch_data['t2_lr']
        mask = batch_data['mask']
        t2_sp, t1_sp = self.model(t2_lr, t1_hr, mask)
        if return_preds:
            return t2_sp, t2_hr

        loss = self.loss_function(t2_sp, t2_hr) + 0.1 * self.loss_function(t1_sp, t1_hr)
        return loss
