import timm 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class STPM(nn.Module):
    def __init__(self, model_name = 'resnet18', layer: list = [0,1,2,3]):
        super(STPM,self).__init__()

        self.teacher = self._create_model(model_name = model_name, pretrained= True)
        self.student = self._create_model(model_name = model_name, pretrained= False)
        self.layer = [str(l+4) for l in layer]                    
        self.mse_loss = nn.MSELoss(reduction="sum")
        
    def _create_model(self, model_name: str, pretrained: bool):
        model = timm.create_model(model_name = model_name, pretrained = pretrained)
        model = nn.Sequential(*list(model.children())[:-2])
        
        if pretrained:
            # teacher required grad False  
            for p in model.parameters():
                p.requires_grad = False 
            model.training = False 
        return model 
    
    def train(self):
        self.student.training = True 
        
    def eval(self):
        self.student.training = False 
            
    def _forward(self, x) -> list:
        t_features = [] 
        s_features = [] 
        
        x_s = x.clone()
        x_t = x.clone() 
        for (t_name, t_module), (s_name,s_module) in zip(self.teacher._modules.items(), self.student._modules.items()):
            x_t = t_module(x_t)
            x_s = s_module(x_s)
            
            if t_name in self.layer:
                t_features.append(x_t)
                s_features.append(x_s)
                
        return t_features, s_features
    
    def _criterion(self, t_features: list, s_features: list):
        total_loss = 0 
        for t,s in zip(t_features, s_features):
            #! Full_base 버전 
            t,s = F.normalize(t, dim=1), F.normalize(s, dim=1)
            layer_loss = torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
            
            #! Anomalib 버전 
            # height, width = t.shape[2:]
            # norm_teacher_features = F.normalize(t)
            # norm_student_features = F.normalize(s)
            # layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)
            
            total_loss += layer_loss
        return total_loss 

    def forward(self, x, only_loss:bool = True):
        t_features, s_features = self._forward(x)
        loss = self._criterion(t_features, s_features)
        
        if only_loss:
            return loss 
        else:
            return loss, [t_features, s_features]
        
            
            
# class SaveHook:
#     def __init__(self):
#         self.outputs = [] 
        
#     def __reset__(self):
#         self.outputs = [] 
        
#     def __call__(self, module, input, output):
#         self.outputs.append(output)

# class STPM(nn.Module):
#     def __init__(self, modelname = 'resnet18', out_dim = 128):
#         super(STPM,self).__init__()
        
#         # create model 
#         self.teacher = timm.create_model(modelname,pretrained=True)
#         self.student = timm.create_model(modelname,pretrained=False)
        
#         # teacher required grad False  
#         for p in self.teacher.parameters():
#             p.requires_grad = False 
            
#         # hook fn 
#         self.teacher_hook = SaveHook()
#         self.student_hook = SaveHook()
                    
#         # register hook 
#         self._register_hook(self.teacher, self.teacher_hook)
#         self._register_hook(self.student, self.student_hook)
        
#     def _register_hook(self, model, hook):
#         for name, module in model._modules.items():
#             if name in ['layer1', 'layer2','layer3','layer4']:
#                 module.register_forward_hook(hook)
        
#     def forward(self, x):            
#         teacher_z = self.teacher(x)
#         student_z = self.student(x)
        
#         return self.teacher_hook.outputs, self.student_hook.outputs