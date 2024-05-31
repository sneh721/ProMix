"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from timm import create_model
from copy import deepcopy

FEAT_DIM = 128


class DualNet(nn.Module):
    def __init__(self, model_name, num_class):
        super().__init__()
        self.net1 = CustomModel(model_name, num_class)
        self.net2 = CustomModel(model_name, num_class)

    def forward(self, x):
        outputs_1 = self.net1(x)
        outputs_2 = self.net2(x)
        outputs_mean = (outputs_1 + outputs_2)/2
        return outputs_mean
    

# Define the init_weights function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        

class CustomModel(nn.Module):
    def __init__(self, model_name, num_class):
        super(CustomModel, self).__init__()
        # transfer all weights including batchnorm? or just linear layer transfer?
        self.model = create_model(model_name=model_name, num_classes=num_class, pretrained=True)
        self.reset_classifier_and_stop_grad()
        
        # copy that maps from dim_in to feat_dim instead of num_classes
        self.train_head = deepcopy(self.model.head)
        self.train_head.fc = nn.Linear(in_features=self.train_head.fc.in_features, 
                                            out_features=FEAT_DIM, bias=True)
        
        self.ph_head = deepcopy(self.model.head)
        
    def forward(self, x, train=False, use_ph=False):
        if train:
            out = self.model.forward_features(x)
            out_linear = self.model.head(out)
            feat_c = self.train_head(out)
            if use_ph:
                out_linear_debias = self.ph_head(out)
                return out_linear, out_linear_debias, F.normalize(feat_c, dim=1)
            else:
                return out_linear, F.normalize(feat_c, dim=1)
        else:
            out = self.model.forward_features(x)
            out_linear = self.model.head(out)
            if use_ph:
                out_linear_debias = self.ph_head(out)
                return out_linear, out_linear_debias
            else:
                return out_linear
    
    def reset_classifier_and_stop_grad(self):
        # Reset weights and biases of the fc block
        self.model.head.apply(init_weights)         # This is the classifier part
        
        # Freeze the backbone - freature extraction layers
        # Do not freeze head
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        
    
def test_custom_model():
    model_name = 'convnext_nano'  # Example TIMM model
    num_classes = 10         # Example number of output classes
    custom_model = CustomModel(model_name, num_classes)
    
    for name, param in custom_model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
        
    # Create a dummy input tensor
    batch_size = 16
    num_channels = 3
    height = 224
    width = 224
    dummy_input = torch.randn(batch_size, num_channels, height, width)
        
    # Forward pass with the dummy input
    custom_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        out, norm = custom_model(dummy_input, train=True, use_ph=False)  
        print(out.shape)
        print(norm.shape) 
        
    with torch.no_grad():
        out, out_debias, norm = custom_model(dummy_input, train=True, use_ph=True) 
        print(out.shape)
        print(out_debias.shape)
        print(norm.shape) 
        
    with torch.no_grad():
        out = custom_model(dummy_input, train=False, use_ph=False) 
        print(out.shape)
        
    with torch.no_grad():
        out, out_debias = custom_model(dummy_input, train=False, use_ph=True)
        print(out.shape)
        print(out_debias.shape)

    print("Feature Extractor Output Shape:", custom_model.train_head.fc.out_features)
    print("Feature Extractor Input Shape:", custom_model.train_head.fc.in_features)
    
    # Print model layers
    for name, layer in custom_model.named_modules():
        print(f"{name}: {layer}")


def main():
    # test_custom_model()
    pass
    
    
if __name__ == "__main__":
    main()