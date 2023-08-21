import timm 
import torch.nn as nn 


class STPM(nn.Module):
    def __init__(self, modelname = 'resnet18', out_dim:int = 128, contrastive:bool = False):
        super(STPM,self).__init__()
        
    
        # create model 
        if contrastive:
            self.teacher = timm.create_model(modelname,pretrained=True, num_classes = out_dim)
        else:
            self.teacher = timm.create_model(modelname,pretrained=True)
        self.student = timm.create_model(modelname,pretrained=False)
        
        # Change student linear for contrastive learning
        if contrastive:
            dim_mlp = self.student.fc.in_features
            self.student.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp,out_dim)
            )
        
        # teacher required grad False  
        for p in self.teacher.parameters():
            p.requires_grad = False 
            
            
    def forward(self, x) -> list:
        t_features = [] 
        s_features = [] 
        
        x_s = x.clone()
        x_t = x.clone() 
        for (t_name, t_module), (s_name,s_module) in zip(self.teacher._modules.items(),self.student._modules.items()):
            x_t = t_module(x_t)
            x_s = s_module(x_s)
            
            if t_name in ['layer1','layer2','layer3','layer4']:
                t_features.append(x_t)
                s_features.append(x_s)
                
                
        return [t_features, s_features, x_t, x_s]
    
    def forward_logit(self, x):
        return self.student(x)
    
            
            
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