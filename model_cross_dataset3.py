from block_cross_dataset3 import *
class RegSeg(nn.Module):
    def __init__(self,  pretrained="", ablate_decoder=False, change_num_classes=False):
        super().__init__()
        self.stem = ConvBnAct(3, 32, 3, 2, 1)

        self.body = RegSegBody()
        self.decoder = eleven_Decoder0(self.body.channels())
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            print(type(dic))
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            print(type(dic))
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x,dataset):
        input_shape=x.shape[-2:]
        x=self.stem(x,dataset)
        x=self.body(x,dataset)
        x=self.decoder(x,dataset)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x