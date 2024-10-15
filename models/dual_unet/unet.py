"""
Off-the-shelf segmentation model from https://github.com/qubvel/segmentation_models.pytorch
"""

import torch, sys
from .unet_module import DualUnet
sys.path.append('your path to the segmentation_models_pytorch package')



class EfficientUNet(torch.nn.Module):
    def __init__(self, nclass, classifier=None, return_feature = False,
                 backbone = 'efficientnet-b2', in_channel = 3, sam=False,
                 norm_type = "bn", act_type = "relu"):

        super(EfficientUNet, self).__init__()
        self.model = DualUnet(
                encoder_name = backbone,
                encoder_weights = None,
                in_channels = in_channel,
                classes = nclass,
                activation = None,
                sam=sam,
                norm_type = norm_type,
                act_type = act_type
                )

        self.return_feature = return_feature
        self.in_channel = in_channel
        

    def get_feature(self, x, enc=False, dual=False, domain=False):
        if dual:
            enc_features = self.model.encoder(x)    
            domain_feature = self.model.encoder_dual(x)    # A set of
            return enc_features, domain_feature
        elif enc:
            enc_features = self.model.encoder(x)           # A set of
            return enc_features
        else:
            domain_feature = self.model.encoder_dual(x)            # A set of
            return domain_feature

    
    def forward_disentangle(self, enc_features, domain_features):
        pass
        # Channel Energy Attention
        

    def forward(self, x, volatile_return_feature = False, enc_features=None, 
                dual=False, D=False, visualization=False, partial=-1):
        
        fusion_way = "s_mask" # "upper", "lower", s_mask
        # Get features
        outputs = {}
        if dual:
            self.nb = x.shape[0] // 4   # [A, A', At, At']
            
            x_enc = x[:3*self.nb]  #torch.cat([x[:self.nb], x[2*self.nb:3*self.nb]], dim=0)
            self.enc_features    = self.get_feature(x_enc, enc=True)
            self.domain_features = self.get_feature(x[:3*self.nb], domain=True)  # A'
            
            feat_a, feat_aprime_mix = [], []
            domain_a = []
            domain_at = []
            
            fusion_dim = -2
            fusion_dim = len(self.enc_features[0]) + fusion_dim
            
            # Cross-  [A, A', At, At']  A^t->A'
            for index, (i, j) in enumerate(zip(self.enc_features, self.domain_features)):
                # Sigmoid
                
                if fusion_way == "s_mask":
                    feat_a.append(j[:self.nb])
                    # sigmoid
                    s_score = torch.sigmoid(i[1*self.nb : 3*self.nb])
                    a = s_score[self.nb:2*self.nb]        * j[2*self.nb:3*self.nb] #2*self.nb:3*self.nb] 
                    b = (1 - s_score[self.nb:2*self.nb] ) * j[self.nb:2*self.nb]
                    
                    # Pure Add
                    
                    
                    
                    feat_aprime_mix.append( a + b )
                    
                    domain_a.append( s_score[:self.nb].clone().detach() * j[self.nb: 2*self.nb])
                    domain_at.append(s_score[self.nb:2*self.nb].clone().detach() * j[2*self.nb:3*self.nb])
                    
                    
                else:
                    if ((index <= fusion_dim) and fusion_way=="upper") or \
                    ((index >= fusion_dim) and fusion_way=="lower"):
                        feat_a.append(i[:self.nb])
                        feat_aprime_mix.append(i[2*self.nb:3*self.nb])
                    else:
                        feat_a.append(j[:self.nb])
                        feat_aprime_mix.append(j[self.nb:2*self.nb])
                    
            self.decoder_a = self.model.decoder_dual(*feat_a)
            outputs['a'] = self.model.reconstruction_head(self.decoder_a)
            
            outputs['domain_a'] = domain_a
            outputs['domain_at'] = domain_at
            
            if D:
                self.decoder_aprime_combine = self.model.decoder_dual(*feat_aprime_mix)
                # self.decoder_a_combine = self.model.decoder_dual(*feat_a_mix)
                outputs['a_prime_mix'] = self.model.reconstruction_head(self.decoder_aprime_combine)

            # outputs['domain_feature'] = 
            
            # self.enc_features = [ i[:self.nb] for i in self.enc_features ]
        elif visualization:
            self.enc_features    = self.get_feature(x, enc=True)
            self.domain_features = self.get_feature(x, domain=True)
            
            fusion_dim = -2
            fusion_dim = len(self.enc_features[0]) + fusion_dim
            
            feat_full = []
            feat_anatomical = []
            feat_domain = []
            
            for index, (i, j) in enumerate(zip(self.enc_features, self.domain_features)):
                if fusion_way == "s_mask":
                        
                    feat_full.append(j)
                    s_score = torch.sigmoid(i)
                    feat_anatomical.append(s_score * j)
                    feat_domain.append( (1 - s_score) * j)
                    
                else:
                    if ((index <= fusion_dim) and fusion_way=="upper") or \
                    ((index >= fusion_dim) and fusion_way=="lower"):  # Anatomical
                        feat_full.append(i)
                        feat_anatomical.append(i)
                        feat_domain.append(torch.zeros_like(j).to(i.device))
                        
                    else:  # Domain
                        feat_full.append(j)
                        feat_anatomical.append(torch.zeros_like(i).to(j.device))
                        feat_domain.append(j)
                    
            self.decoder_a = self.model.decoder_dual(*feat_full)
            outputs['full'], outputs['full_comp'] = self.model.reconstruction_head(self.decoder_a, return_comp=True)
            self.decoder_a = self.model.decoder_dual(*feat_anatomical)
            outputs['anatomical'], outputs['anatomical_comp'] = self.model.reconstruction_head(self.decoder_a, return_comp=True)
            self.decoder_a = self.model.decoder_dual(*feat_domain)
            outputs['domain'], outputs['domain_comp'] = self.model.reconstruction_head(self.decoder_a, return_comp=True)
            
        else:
            
            self.enc_features = self.get_feature(x, enc=True)

        # Forward features
        self.decoder_output = self.model.decoder(*self.enc_features)
        masks = self.model.segmentation_head(self.decoder_output)
        outputs['enc_features'] = self.enc_features
        outputs['masks'] = masks

        return outputs


    def forward_domain(self, domain_feature):
        self.decoder_dual_output = self.model.decoder_dual(*domain_feature)
        return self.model.reconstruction_head(self.decoder_dual_output)






def efficient_unet(**kwargs):
    return EfficientUNet(**kwargs)



