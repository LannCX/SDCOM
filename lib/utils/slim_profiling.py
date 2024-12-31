import torch.nn as nn
model_mac_hooks = []


def module_mac(self, input, output):
    if isinstance(input[0], tuple):
        if isinstance(input[0][0], list):
            ins = input[0][0][3].size()
        else:
            ins = input[0][0].size()
    else:
        ins = input[0].size()
    if isinstance(output, tuple):
        if isinstance(output[0], list):
            outs = output[0][3].size()
        else:
            outs = output[0].size()
    else:
        outs = output.size()
    
    if isinstance(self, (nn.Conv2d, nn.ConvTranspose2d)):
        try:
            # Dynamic Conv
            # print(type(self.running_inc), type(self.running_outc), type(self.running_kernel_size), type(outs[2]), type(self.running_groups))
            self.running_flops = (self.running_inc * self.running_outc *
                                self.running_kernel_size * self.running_kernel_size *
                                outs[2] * outs[3] / self.running_groups)
            # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
        except AttributeError as e:
            # Normal Conv
            self.running_flops = self.in_channels*self.out_channels*self.kernel_size[0]*self.kernel_size[1]*outs[2]*outs[3]/self.groups
    elif isinstance(self, nn.Linear):
        try:
            self.running_flops = self.running_inc * self.running_outc
        except AttributeError as e:
            self.running_flops = self.out_features * self.in_features
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    # elif isinstance(self, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
    #     # NOTE: this function is correct only when stride == kernel size
    #     self.running_flops = self.running_inc * ins[2] * ins[3]
    #     # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    
    return


def add_mac_hooks(m):
    global model_mac_hooks
    model_mac_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_mac(
            m, input, output)))


def remove_mac_hooks():
    global model_mac_hooks
    for h in model_mac_hooks:
        h.remove()
    model_mac_hooks = []


def add_flops(model):
    flops = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.Linear):
            flops += getattr(m, 'running_flops', 0)

    return flops


if __name__ == '__main__':
    main()
