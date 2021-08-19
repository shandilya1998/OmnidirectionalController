import torch
from constants import params
from collections import OrderedDict

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'cuda'

def complex_relu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.relu(x),
            torch.nn.functional.relu(y)
        ], - 1
    )
    return out

def complex_elu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.elu(x),
            torch.nn.functional.elu(y)
        ], - 1
    )
    return out



def complex_tanh(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    denominator = torch.cosh(2*x) + torch.cos(2*y)
    x = torch.sin(2*x) / denominator
    y = torch.sinh(2*y) / denominator
    out = torch.cat([x, y], -1)
    return out

def apply_complex(fr, fi, input, dtype = torch.float32):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            fr(x) - fi(y),
            fr(y) + fi(x)
        ], -1
    )
    return out

class ComplexReLU(torch.nn.Module):

     def forward(self,input):
         return complex_relu(input)

class ComplexELU(torch.nn.Module):

     def forward(self,input):
         return complex_elu(input)

class ComplexTanh(torch.nn.Module):

    def forward(self, input):
        return complex_tanh(input)

class ComplexPReLU(torch.nn.Module):
    def __init__(self):
        super(ComplexPReLU, self).__init__()
        self.prelu_r = torch.nn.PReLU()
        self.prelu_i = torch.nn.PReLU()

    def forward(self, input):
        size = input.shape[-1]//2
        x, y = torch.split(input, size, -1)
        out = torch.cat([self.prelu_r(x), self.prelu_i(y)], -1)
        return out

class ComplexLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features)
        self.fc_i = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ComplexConvTranspose2d(torch.nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)


    def forward(self,input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

class ComplexConv2d(torch.nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input):
        return apply_complex(self.conv_r, self.conv_i, input)

class Hopf(torch.nn.Module):
    def __init__(self, params):
        super(Hopf, self).__init__()
        self.params = params
        self.dt = torch.Tensor([self.params['dt']]).type(FLOAT)

    def forward(self, z, omega, mu):
        units_osc = z.shape[-1]
        x, y = torch.split(z, units_osc // 2, -1)
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y,x)
        delta_phi = self.dt * omega
        phi = phi + delta_phi
        r = r + self.dt * (mu - r ** 2) * r
        z = torch.cat([x, y], -1)
        return z

class ParamNet(torch.nn.Module):
    def __init__(self,
        params,
    ):
        super(ParamNet, self).__init__()
        self.params = params
        motion_seq = []
        input_size =  params['motion_state_size']
        output_size_motion_state_enc = None
        for i, units in enumerate(params['units_motion_state']):
            motion_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            motion_seq.append(torch.nn.PReLU())
            input_size = units
            output_size_motion_state_enc = units
        self.motion_state_enc = torch.nn.Sequential(
            *motion_seq
        )

        omega_seq = []
        for i, units in enumerate(params['units_omega']):
            omega_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            omega_seq.append(torch.nn.PReLU())
            input_size = units

        omega_seq.append(torch.nn.Linear(
            input_size,
            params['units_osc']
        ))
        omega_seq.append(torch.nn.ReLU())

        self.omega_dense_seq = torch.nn.Sequential(
            *omega_seq
        )

        mu_seq = []
        for i, units in enumerate(params['units_mu']):
            mu_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            mu_seq.append(torch.nn.PReLU())
            input_size = units

        mu_seq.append(torch.nn.Linear(
            input_size,
            params['units_osc']
        ))
        self.out_relu = torch.nn.ReLU()

        self.mu_dense_seq = torch.nn.Sequential(
            *mu_seq
        )

    def forward(self, desired_motion):
        x = self.motion_state_enc(desired_motion)
        omega = self.omega_dense_seq(x)
        mu = self.out_relu(torch.cos(self.mu_dense_seq(x)))
        return omega, mu

class Controller(torch.nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        input_size = params['input_size_low_level_control']
        output_mlp_seq = []
        for i, units in enumerate(params['units_output_mlp']):
            output_mlp_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            output_mlp_seq.append(torch.nn.PReLU())
            input_size = units

        output_mlp_seq.append(torch.nn.Linear(
            input_size,
            params['cpg_param_size']
        ))

        self.output_mlp = torch.nn.Sequential(
            *output_mlp_seq
        )

    def forward(self, ob):
        return self.output_mlp(ob)


class ControllerV2(torch.nn.Module):
    def __init__(self):
        super(ControllerV2, self).__init__()
        input_size = params['input_size_low_level_control']
        """
        output_mlp_seq = []
        for i, units in enumerate(params['units_output_mlp']):
            output_mlp_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            output_mlp_seq.append(torch.nn.PReLU())
            input_size = units

        output_mlp_seq.append(torch.nn.Linear(
            input_size,
            params['cpg_param_size']
        ))

        self.output_mlp = torch.nn.Sequential(
            *output_mlp_seq
        )
        """

        self.encoder = torch.nn.Linear(params['cpg_param_size'], input_size)
        self.decoder = torch.nn.Linear(input_size, params['cpg_param_size'])

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z
