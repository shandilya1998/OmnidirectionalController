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

class Hopf(torch.nn.Module):
    def __init__(self, params):
        super(Hopf, self).__init__()
        self.params = params
        self.dt = torch.Tensor([self.params['dt']]).type(FLOAT)
        self.arange = torch.arange(0, self.params['units_osc'], 1.0).type(FLOAT)

    def forward(self, z, omega, mu):
        units_osc = z.shape[-1]
        x, y = torch.split(z, units_osc // 2, -1)
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y,x)
        delta_phi = self.dt * omega * self.arange
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
            1
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
        self.param_net = ParamNet(params)
        self.hopf = Hopf(params)

        """
        robot_state_enc_seq = OrderedDict()
        input_size = params['robot_state_size']
        for i, units in enumerate(params['units_robot_state']):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.PReLU()
            input_size = units
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)
        """
        output_mlp_seq = []
        for i, units in enumerate(params['units_omega']):
            output_mlp_seq.append(ComplexLinear(
                input_size,
                units
            ))
            output_mlp_seq.append(torch.nn.PReLU())
            input_size = units

        output_mlp_seq.append(ComplexLinear(
            input_size,
            params['action_dim']
        ))

        self.out_tanh = torch.nn.Tanh()

        self.output_mlp = torch.nn.Sequential(
            *output_mlp_seq
        )

    def forward(self, ob, z):
        desired_goal, achieved_goal, observation = ob
        omega, mu = self.param_net(desired_goal)
        """
        z_r = self.robot_state_enc(torch.cat([achieved_goal, observation], -1))
        z_i = torch.zeros_like(z_r)
        """
        z = self.hopf(z, omega, mu)
        #out = self.output_mlp(z + torch.cat([z_r, z_i], -1))
        out = self.output_mlp(z)
        x, y = torch.split(out, [params['action_dim'], params['action_dim']], -1)
        x = params['max_action'] * self.out_tanh(x)
        out = torch.cat([x, z], -1)
        """
            get x and concat with z
        """
        return out

