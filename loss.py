import torch
import torch.nn as nn


class SpatialFrequencyLoss_L1(nn.Module):
    def __init__(self, fft_weight=0.01):
        """
        SpatialFrequencyLoss class that combines L1 loss, amplitude loss, and phase loss.
        Args:
            fft_weight (float): Weight applied to the FFT-based losses (amplitude and phase).
        """
        super(SpatialFrequencyLoss_L1, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.fft_weight = fft_weight

    def compute_fft_features(self, x):
        """
        Compute FFT features (amplitude and phase).
        Returns:
            tuple: Amplitude and phase of the FFT-transformed input.
        """
        x_fft = torch.fft.rfft2(x, norm='backward')
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return amplitude, phase

    def forward(self, outputs, targets):
        # L1 loss
        l1_loss = self.l1_loss(outputs, targets)

        # FFT-based losses
        output_ampl, output_phase = self.compute_fft_features(outputs)
        target_ampl, target_phase = self.compute_fft_features(targets)

        ampl_loss = self.l1_loss(output_ampl, target_ampl)
        pha_loss = self.l1_loss(output_phase, target_phase)

        # Combine losses
        combined_loss = l1_loss + self.fft_weight * (ampl_loss + pha_loss)
        # print(l1_loss.item(), ampl_loss.item(), pha_loss.item(), combined_loss.item())
        return combined_loss


class SpatialFrequencyLoss_MSE(nn.Module):
    def __init__(self, fft_weight=0.01):
        """
        SpatialFrequencyLoss class that combines L1 loss, amplitude loss, and phase loss.
        Args:
            fft_weight (float): Weight applied to the FFT-based losses (amplitude and phase).
        """
        super(SpatialFrequencyLoss_MSE, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.fft_weight = fft_weight

    def compute_fft_features(self, x):
        """
        Compute FFT features (amplitude and phase).
        Returns:
            tuple: Amplitude and phase of the FFT-transformed input.
        """
        x_fft = torch.fft.rfft2(x, norm='backward')
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return amplitude, phase

    def forward(self, outputs, targets):
        # L1 loss
        mse_loss = self.mse_loss(outputs, targets)

        # FFT-based losses
        output_ampl, output_phase = self.compute_fft_features(outputs)
        target_ampl, target_phase = self.compute_fft_features(targets)

        ampl_loss = self.mse_loss(output_ampl, target_ampl)
        pha_loss = self.mse_loss(output_phase, target_phase)

        # Combine losses
        combined_loss = mse_loss + self.fft_weight * (ampl_loss + pha_loss)
        # print(mse_loss.item(), ampl_loss.item(), pha_loss.item(), combined_loss.item())
        return combined_loss


class SpatialFrequencyLoss_Next(nn.Module):
    def __init__(self, fft_weight=0.01):
        """
        SpatialFrequencyLoss class that combines L1 loss, amplitude loss, and phase loss.
        Args:
            fft_weight (float): Weight applied to the FFT-based losses (amplitude and phase).
        """
        super(SpatialFrequencyLoss_Next, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.fft_weight = fft_weight

    def compute_fft_features(self, x):
        """
        Compute FFT features (amplitude and phase).
        Returns:
            tuple: Amplitude and phase of the FFT-transformed input.
        """
        x_fft = torch.fft.rfft2(x, norm='backward')
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return amplitude, phase

    def forward(self, outputs, targets):
        # space
        mse_loss = self.mse_loss(outputs, targets)

        # FFT-based losses
        output_ampl, output_phase = self.compute_fft_features(outputs)
        target_ampl, target_phase = self.compute_fft_features(targets)

        ampl_loss = self.l1_loss(output_ampl, target_ampl)
        pha_loss = self.l1_loss(output_phase, target_phase)

        # Combine losses
        combined_loss = mse_loss + self.fft_weight * (ampl_loss + pha_loss)
        # print(mse_loss.item(), ampl_loss.item(), pha_loss.item(), combined_loss.item())
        return combined_loss


class LossNotIntegrated(nn.Module):
    def __init__(self, fft_weight=0.01):
        """
        SpatialFrequencyLoss class that combines L1 loss, amplitude loss, and phase loss.
        Args:
            fft_weight (float): Weight applied to the FFT-based losses (amplitude and phase).
        """
        super(LossNotIntegrated, self).__init__()
        print('Caution: Loss is not intergrated in loss.py, but running in wrapper.py')

    def forward(self, outputs, targets):
        raise Exception('Class LossNotIntergrated should not be called.')


if __name__ == '__main__':
    pd_hr = torch.randn(1, 1, 200, 200)
    t2_lr = torch.randn(1, 1, 200, 200)
    loss_fn = SpatialFrequencyLoss_Next()
    loss = loss_fn(pd_hr, t2_lr)
    print(loss)
