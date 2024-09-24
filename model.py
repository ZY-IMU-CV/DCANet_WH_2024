from blocks import *
from block_decoder import *
from block_decoder2 import *
from block_FAM import *
from block_FAM2 import *
from blocks_wh import *
from blocks_wh_mobileone import *
from blocks_whpaper import *
from block_decoderpaper import *


class RegSeg(nn.Module):
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False, change_num_classes=False):
        super().__init__()
        dilations1 = 1
        attention = 'se'
        self.stem = ConvBnAct(3, 32, 3, 2, 1)
        body_name, decoder_name = name.split("_")
        if "eleven221418" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[1, 4]] + 2 * [[1, 8]])
        elif "eleven224488" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "nine2248" == body_name:
            self.body = RegSegBody(2 * [[2]] + [[4]] + [[8]] )
        elif "ten22448" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + [[8]] )
        elif "twelve22448816" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2*[[8]]+[[16]] )
        elif "thirteen2244881616" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2*[[8]]+ 2*[[16]] )
        elif "twelvedown320" == body_name:
            self.body = RegSegBody2(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "twelvedown321" == body_name:
            self.body = RegSegBody3(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "ten1" == body_name:
            self.body = RegSegBody_1()
        elif "ten2" == body_name:
            self.body = RegSegBody_2()
        elif "ten3" == body_name:
            self.body = RegSegBody_3()
        elif "ten4" == body_name:
            self.body = RegSegBody_4()
        elif "ten5" == body_name:
            self.body = RegSegBody_5()
        elif "ten6" == body_name:
            self.body = RegSegBody_6()
        elif "fourteen1" == body_name:
            self.body = RegSegBody_7()
        elif "fourteen2" == body_name:
            self.body = RegSegBody_8()
        elif "fourteen3" == body_name:
            self.body = RegSegBody_9()
        elif "fourteen4" == body_name:
            self.body = RegSegBody_10()
        elif "eighteen1" == body_name:
            self.body = RegSegBody_11()
        elif "eighteen2" == body_name:
            self.body = RegSegBody_12()
        elif "eighteen3" == body_name:
            self.body = RegSegBody_13()
        elif "eighteen4" == body_name:
            self.body = RegSegBody_14()
        elif "regtest" == body_name:
            self.body = RegSegBody_test()
        elif "tentwo1" == body_name:
            self.body = RegSegBody_15()  ##RAFNet CL  final
        elif "tentwo1selfattention" == body_name:
            self.body = RegSegBody_15_selfattention()  ##RAFNet CL  final
        elif "LightSegE0" == body_name:        # final ICANet   CL
            self.encoder = LightSeg_Encoder_0()
        elif "tentwo1aspp" == body_name:
            self.body = RegSegBody_15aspp()
        elif "tentwo1320" == body_name:
            self.body = RegSegBody_152()
        elif "tentwo1short" == body_name:
            self.body = RegSegBody_153()
        elif "tentwo2" == body_name:
            self.body = RegSegBody_16()
        elif "tentwo3" == body_name:
            self.body = RegSegBody_17()
        elif "tentwo3Y" == body_name:
            self.body = RegSegBody_17Y()
        elif "tentwo3D" == body_name:
            self.body = RegSegBody_17D()
        elif "tentwo4" == body_name:
            self.body = RegSegBody_18()
        elif "tentwo5" == body_name:
            self.body = RegSegBody_19()
        elif "tentwo6" == body_name:
            self.body = RegSegBody_20()
        elif "tentwo7" == body_name:
            self.body = RegSegBody_21()
        elif "tentwo8" == body_name:
            self.body = RegSegBody_22()
        elif "tentwo9" == body_name:
            self.body = RegSegBody_23()
        elif "tentwo10" == body_name:
            self.body = RegSegBody_24()
        elif "tentwo11" == body_name:
            self.body = RegSegBody_25()
        elif "tentwo12" == body_name:
            self.body = RegSegBody_26()
        elif "tentwo13" == body_name:
            self.body = RegSegBody_27()
        elif "tentwo14" == body_name:
            self.body = RegSegBody_28()
        elif "tentwo15" == body_name:
            self.body = RegSegBody_29()
        elif "tentwo16" == body_name:
            self.body = RegSegBody_30()
        elif "testshort"==body_name:
            self.body = RegSegBody_addallshort()
        elif "testrepvgg1"==body_name:
            self.body = RegSegBody_repvgg()
        elif "tentwo3wh"==body_name:
            self.body = RegSegBody_17_wh()
        elif "tentwo3wh2"==body_name:
            self.body = RegSegBody_17_wh2()
        elif "tentwo3wh3"==body_name:
            self.body = RegSegBody_17_wh3()
        elif "tentwo3wh4"==body_name:
            self.body = RegSegBody_17_wh4()
        elif "tentwo3wh5"==body_name:
            self.body = RegSegBody_17_wh4_add()
        elif "tentwo3wh6"==body_name:
            self.body = RegSegBody_17_wh5_d4()
        elif "tentwo3wh7"==body_name:
            self.body = RegSegBody_17_wh6_d3()
        elif "tentwo3wh8"==body_name:
            self.body = RegSegBody_17_wh7_d6()
        elif "tentwo3wh9"==body_name:
            self.body = RegSegBody_17_wh8_d4()
        elif "tentwo3wh10"==body_name:
            self.body = RegSegBody_17_wh10_d4()
        elif "tentwo3wh11"==body_name:
            self.body = RegSegBody_17_wh11()
        elif "tentwo3wh12"==body_name:
            self.body = RegSegBody_17_wh12()
        elif "tentwo3wh13"==body_name:
            self.body = RegSegBody_17_wh13()
        elif "tentwo3wh14"==body_name:
            self.body = RegSegBody_17_wh14()
        elif "tentwo3wh15"==body_name:
            self.body = RegSegBody_17_wh15()
        elif "tentwo3wh16"==body_name:
            self.body = RegSegBody_17_wh16()
        elif "tentwo3wh17"==body_name:
            self.body = RegSegBody_17_wh17()
        elif "tentwo3wh18"==body_name:
            self.body = RegSegBody_17_wh18_add()
        elif "tentwo3wh19"==body_name:
            self.body = RegSegBody_17_wh19_self_attention()
        elif "tentwo3wh20"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_cat()
        elif "tentwo3wh21"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_2()
        elif "tentwo3wh22"==body_name:
            self.body = RegSegBody_17_wh22()
        elif "tentwo3wh23"==body_name:
            self.body = RegSegBody_17_wh23()
        elif "tentwo3wh24"==body_name:
            self.body = RegSegBody_17_wh24()
        elif "tentwo3wh25"==body_name:
            self.body = RegSegBody_17_wh25()
        elif "tentwo3wh26"==body_name:
            self.body = RegSegBody_17_wh26()
        elif "tentwo3wh27"==body_name:
            self.body = RegSegBody_17_wh27()
        elif "tentwo3wh28"==body_name:
            self.body = RegSegBody_17_wh28()
        elif "tentwo3wh29"==body_name:
            self.body = RegSegBody_17_wh29()
        elif "tentwo3wh30"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_3()
        elif "tentwo3wh31"==body_name:
            self.body = RegSegBody_17_wh31()
        elif "tentwo3wh32"==body_name:
            self.body = RegSegBody_17_wh32()
        elif "tentwo3wh32"==body_name:
            self.body = RegSegBody_17_wh33()
        elif "exp53" == body_name:
            self.body = RegSegBody2wh([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        elif "exp48" == body_name:
            self.body = RegSegBody3wh([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        elif "exp48selfattention" == body_name:
            self.body = RegSegBody3wh_selfattention([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]]) #regseg selfattention
        elif "exp54wh" == body_name:
            self.body = RegSegBodywh1([[1], [1, 2]] + 4 * [[1, 4]] + 8 * [[1, 14]])
        elif "tentwo3wh33"==body_name:
            self.body = RegSegBody_17_wh33()
        elif "tentwo3wh34"==body_name:
            self.body = RegSegBody_17_wh34()
        elif "tentwo3wh35"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_cat_stage3()
        elif "tentwo3wh36"==body_name:
            self.body = RegSegBody_17_wh36()
        elif "tentwo3wh37"==body_name:
            self.body = RegSegBody_17_wh37()
        elif "tentwo3wh38"==body_name:
            self.body = RegSegBody_17_wh38()
        elif "tentwo3wh39"==body_name:
            self.body = RegSegBody_17_wh39()
        elif "tentwo3wh40"==body_name:
            self.body = RegSegBody_17_wh40()
        elif "tentwo3wh41"==body_name:
            self.body = RegSegBody_17_wh41()
        elif "tentwo3wh17stage2block"==body_name:
            self.body = RegSegBody_17_wh17_satge2block()
        elif "tentwo3wh42"==body_name:
            self.body = RegSegBody_17_wh42()
        elif "tentwo3wh43"==body_name:
            self.body = RegSegBody_17_wh43()
        elif "tentwo3wh17d1"==body_name:
            self.body = RegSegBody_17_wh_dilation_all1()
        elif "tentwo3wh17d1234666"==body_name:
            self.body = RegSegBody_17_wh_dilation_1234666()
        elif "tentwo3wh17d1234468"==body_name:
            self.body = RegSegBody_17_wh_dilation_1234468()
        elif "tentwo3whnum334"==body_name:
            self.body = RegSegBody_17_wh17num334()
        elif "tentwo3whnum345"==body_name:
            self.body = RegSegBody_17_wh17num345()
        elif "tentwo3whnum456"==body_name:
            self.body = RegSegBody_17_wh17num456()
        elif "tentwo3whnum555"==body_name:
            self.body = RegSegBody_17_wh17num555()
        elif "tentwo3whnum467"==body_name:
            self.body = RegSegBody_17_wh17num467()
        elif "tentwo3whnum578"==body_name:
            self.body = RegSegBody_17_wh17num578()
        elif "tentwo3whnum1467"==body_name:
            self.body = RegSegBody_17_wh17num1467()
        elif "tentwo3whnum467block1"==body_name:
            self.body = RegSegBody_17_wh17num467block1()
        elif "tentwo3whnum467block2" == body_name:
            self.body = RegSegBody_17_wh17num467block2()
        elif "tentwo3whnum467block4" == body_name:
            self.body = RegSegBody_17_wh17num467block4()
        elif "tentwo3whnum467block5" == body_name:
            self.body = RegSegBody_17_wh17num467block5()
        elif "tentwo3whnum467block6" == body_name:
            self.body = RegSegBody_17_wh17num467block6()
        elif "tentwo3whnum467channel5" == body_name:
            self.body = RegSegBody_17_wh17num467channel5()
        elif "tentwo3whnum467channel3" == body_name:
            self.body = RegSegBody_17_wh17num467channel3()
        elif "tentwo3whnum467channel1" == body_name:
            self.body = RegSegBody_17_wh17num467channel1()
        elif "tentwo3whnum467channel2" == body_name:
            self.body = RegSegBody_17_wh17num467channel2()
        elif "tentwo3whnum467block8" == body_name:
            self.body = RegSegBody_17_wh17num467block8()
        elif "tentwo3whnum467block7" == body_name:
            self.body = RegSegBody_17_wh17num467block7()
        elif "tentwo3whnum467d2" == body_name:
            self.body = RegSegBody_17_wh17num467d2()
        elif "tentwo3whnum467d2244466" == body_name:
            self.body = RegSegBody_17_wh17num467d2244466()
        elif "tentwo3whnum467d4" == body_name:
            self.body = RegSegBody_17_wh17num467d4()
        elif "tentwo3whnum467d6" == body_name:
            self.body = RegSegBody_17_wh17num467d6()
        elif "tentwo3whnum467d2224468" == body_name:
            self.body = RegSegBody_17_wh17num467d2224468()
        elif "tentwo3whnum444"==body_name:
            self.body = RegSegBody_17_wh17num444()
        elif "tentwo3whnum566"==body_name:
            self.body = RegSegBody_17_wh17num566()
        elif "tentwo3whnum467d4cat17"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat17()
        elif "tentwo3whnum467d4cat147"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat147()
        elif "tentwo3whnum467d4cat1357"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat1357()
        elif "tentwo3whnum467d4cat2467"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467()#     DCANET    ----final
        elif "tentwo3whnum467d4cat2467channel1"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_channel1()#     DCANET    ----final-channel1-(32,64,64,128,256)
        elif "tentwo3whnum467d4cat2467channel2"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_channel2()#     DCANET    ----final-channel1-(32,64,128,128,256)
        elif "tentwo3whnum467d4cat2467channel4"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_channel4()#     DCANET    ----final-channel1-(32,64,128,256,512)
        elif "tentwo3whnum467d4cat2467channel5"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_channel5()#     DCANET    ----final-channel1-(32,64,128,256,1024)
        elif "tentwo3whnum345d4cat1245"==body_name:
            self.body = RegSegBody_17_wh17num345d4cat1245()#     DCANET    ----final-block(3,4,5) cat 1245
        elif "tentwo3whnum467d4cat2467Yblock"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_Yblock()#     用Y block替换
        elif "tentwo3whnum467d4cat2467RCDblock"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_RCDblock()#     用RCD block替换
        elif "tentwo3whnum467d4cat2467IDSblock"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467_IDSblock()#    用IDS block替换
        elif "tentwo3whnum665"==body_name:
            self.body = RegSegBody_17_wh17num665()
        elif "tentwo3whnum11244"==body_name:
            self.body = RegSegBody_17_wh17num11244()
        elif "tentwo3whnum11357"==body_name:
            self.body = RegSegBody_17_wh17num11357()
        elif "tentwo3whnum11467"==body_name:
            self.body = RegSegBody_17_wh17num11467()
        elif "tentwo3whnum01467"==body_name:
            self.body = RegSegBody_17_wh17num01467()
        elif "tentwo3whnum01345"==body_name:
            self.body = RegSegBody_17_wh17num01345()
        elif "tentwo3whnum11345"==body_name:
            self.body = RegSegBody_17_wh17num11345()
        elif "tentwo3whnum467d28cat"==body_name:
            self.body = RegSegBody_17_wh17num467d28cat()
        elif "tentwo3whnum467d4add2467"==body_name:
            self.body = RegSegBody_17_wh17num467d4add2467()
        elif "tentwo3whnum11467d4cat2467"==body_name:
            self.body = RegSegBody_17_wh17num11467d4cat2467()
        elif "tentwo3whnum689d4cat24679"==body_name:
            self.body = RegSegBody_17_wh17num689d4cat24679()
        elif "tentwo3whnum7910d4cat2467910"==body_name:
            self.body = RegSegBody_17_wh17num7910d4cat2467910()
        elif "tentwo3whnum689d4cat24679512"==body_name:
            self.body = RegSegBody_17_wh17num689d4cat24679512()
        elif "tentwo3whnum3689d4cat24679"==body_name:
            self.body = RegSegBody_17_wh17num3689d4cat24679()
        elif "tentwo3whnum467vit"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_vit()
        elif "tentwo3whnum234vit"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_vit()
        elif "tentwo3whnum467selfvit"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit()
        elif "tentwo3whnum7912"==body_name:
            self.body = RegSegBody_17_wh17num7912d4cat2467912()
        elif "tentwo3whnum467d8cat2467"==body_name:
            self.body = RegSegBody_17_wh17num467d8cat2467()
        elif "tentwo3whnum467selfvits6"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s6()
        elif "tentwo3whnum467selfvits6b2"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s6_2()
        elif "tentwo3whnum467selfvits6b3"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s6_3()
        elif "tentwo3whnum467d10cat2467"==body_name:
            self.body = RegSegBody_17_wh17num467d10cat2467()
        elif "tentwo3whnum7912d8"==body_name:
            self.body = RegSegBody_17_wh17num7912d8cat2467912()
        elif "tentwo3whnum7913"==body_name:
            self.body = RegSegBody_17_wh17num7912d4cat2467913()
        elif "tentwo3regsegnum467d4cat2467"==body_name:
            self.body = RegSegBody_17_whregsegnum467d4cat2467()
        elif "tentwo3whnum467d4cat2467sb"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467sb()
        elif "tentwo3whnum467d4cat2467sbdw1"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467sbdw1()
        elif "tentwo3whnum467d4cat2467dbdw1"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467sbdw1()
        elif "tentwo3whnum467d4cat2467sbdwhou1"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467sbdwhou1()
        elif "tentwo3whnum467d4cat2467db1dw"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467db1dw()
        elif "tentwo3whnum467d4cat2467posi"==body_name:
            self.body = RegSegBody_17_wh17num467d4cat2467posi()
        elif "tentwo3whnum467d1cat2467"==body_name:
            self.body = RegSegBody_17_wh17num467d1cat2467()
        elif "tentwo3whnum467d2244466cat2467"==body_name:
            self.body = RegSegBody_17_wh17num467d2244466cat2467()
        elif "tentwo3whnum7912posi"==body_name:
            self.body = RegSegBody_17_wh17num7912d4cat2467912posi()
        elif "tentwo3whnum467selfvits6right"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s613qk()
        elif "tentwo3whnum467selfvits6b2right"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s6_2right()
        elif "tentwo3whnum467selfvits5b2right"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right()
        elif "tentwo3whnum467selfvits5"==body_name:# 只经过s5
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5()
        elif "tentwo3whnum467selfvits6b31q3kv"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s6_2right_1q3kv()
        elif "tentwo3whnum467selfvits5b7s6b2cat12"==body_name:
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5b7s6b2_cat12()
        elif "tentwo3whnum334selfvits5b2right"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3342()
        elif "tentwo3whnum334selfvits5b2rightd6"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3342d6()
        elif "tentwo3whnum234selfvits5b2right"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342()
        elif "tentwo3whnum234selfvits5b2rightdd4"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342dd4()
        elif "tentwo3whnum234selfvits5b2rightdd3"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342dd3()
        elif "tentwo3whnum234selfvits5b2rightddlight2"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight2()
        elif "tentwo3whnum234selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样----final
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1()
        elif "tentwo3whnum234selfvits5b2rightddlight3"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight3()
        elif "tentwo3whnum333selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3332ddlight1()
        elif "tentwo3whnum135selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b1352ddlight1()
        elif "tentwo3whnum2343selfvits5b3rightddlight1"==body_name:# 先经过s6下采样后再经过上采样 stage6 block=3
            self.body = RegSegBody_17_wh19_self_attention_selfvit2_s5_2right_b2342ddlight1()
        elif "tentwo3whnum235selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2352ddlight1()
        elif "tentwo3whnum345selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3452ddlight1()
        elif "tentwo3whnum234selfvits5b2rightddlight1ASPP"==body_name:# 先经过s6下采样后再经过上采样----ASPP
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_ASPP()
        elif "tentwo3whnum234selfvits5b2rightddlight1NL"==body_name:# NON local
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1NL()
        elif "tentwo3whnum234selfvits5b2rightddlight1last4stage"==body_name:# 将后四个stage输入到decoder
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_last4stage()
        elif "tentwo3whnum234selfvits5b2rightddlight1DNL"==body_name:# 先经过s6下采样后再经过上采样---D_non_local
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_DNL()
        elif "tentwo3whnum234selfvits5b2rightddlight1PPM"==body_name:# 先经过s6下采样后再经过上采样----+PPM
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_PPM()
        elif "tentwo3whnum234selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样----通道数消融实验
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1()
        elif "tentwo3whnum01234selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样----(0,1,2,3,4)
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b012342ddlight1()
        elif "tentwo3whnum432selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样----(4,3,2,2)
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b4322ddlight1()
        elif "tentwo3whnum234ddlight1"==body_name:# 先经过s6下采样后再经过上采样----不添加自注意力
            self.body = RegSegBody_17_wh19_s5_2right_b2342ddlight1()
        elif "tentwo3whnum234ddlight1DA"==body_name:# 先经过s6下采样后再经过上采样----添加自注意力DA
            self.body = RegSegBody_17_wh19_s5_2right_b2342ddlight1DA()
        elif "tentwo3whnum11234selfvits5b2rightddlight1"==body_name:# 先经过s6下采样后再经过上采样----(0,1,2,3,4)
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b112342ddlight1()
        elif "tentwo3whnum234selfvits5b2rightddlight1psaa"==body_name:# 论文中的PSA-（a）
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_psaa()
        elif "tentwo3whnum234selfvits5b2rightddlight1ASPPpsab"==body_name:# 先经过s6下采样后再经过上采样----ASPP
            self.body = RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_ASPP_psab()
        else:
            raise NotImplementedError()

        if "decoder11" ==decoder_name:
            self.decoder = eleven_Decoder0(num_classes, self.body.channels())
        elif "decoder12" ==decoder_name:
            self.decoder = eleven_Decoder1(num_classes, self.body.channels())
        elif "decoder13" ==decoder_name:
            self.decoder = eleven_Decoder2(num_classes, self.body.channels())
        elif "LRASPP" ==decoder_name:
            self.decoder = LRASPP(num_classes, self.body.channels())
        elif "decoderdown32" == decoder_name:
            self.decoder = down32_Decoder0(num_classes, self.body.channels())
        elif "down32ASPP" == decoder_name:
            self.decoder = down32_Decoder1(num_classes, self.body.channels())
        elif "BiFPN32" == decoder_name:
            self.decoder = BiFPN([48, 128, 256, 256,19])
        elif "STDC32" == decoder_name:
            self.decoder = STDC_docker()
        elif "down32test" == decoder_name:
            self.decoder = down32_test(num_classes, self.body.channels())
        elif "EASPP" == decoder_name:
            self.decoder = eASPP_deccoder(num_classes, self.body.channels())
        elif "EASPP2" ==decoder_name:
            self.decoder = eASPP_deccoder2(num_classes, self.body.channels())
        elif "EASPP3" ==decoder_name:
            self.decoder = eASPP_deccoder3(num_classes, self.body.channels())
        elif "EASPP4" ==decoder_name:
            self.decoder = eASPP_deccoder4(num_classes, self.body.channels())
        elif "EASPP5" ==decoder_name:
            self.decoder = eASPP_deccoder5(num_classes, self.body.channels())
        elif "down32cat" == decoder_name:
            self.decoder = down32_Decoder_cat(num_classes, self.body.channels())
        elif "down32sum" == decoder_name:
            self.decoder = down32_Decoder_sum(num_classes, self.body.channels())
        elif "down3264" == decoder_name:
            self.decoder = down32_Decoder64(num_classes, self.body.channels())
        elif "down32128" == decoder_name:
            self.decoder = down32_Decoder128(num_classes, self.body.channels())
        elif "decoderproccess64" == decoder_name:
            self.decoder = down32_Decoderprocess64(num_classes, self.body.channels())
        elif "decoder1x1" == decoder_name:
            self.decoder = down32_Decoder1x1(num_classes, self.body.channels())
        elif "decoderban" == decoder_name:
            self.decoder = down32_Decoderban(num_classes, self.body.channels()) 
        elif "decoderFAM" == decoder_name:
            self.decoder = decoder_FAM(num_classes, self.body.channels())
        elif "decoderFAMfinish" == decoder_name:
            self.decoder = UperNetAlignHead(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish2" == decoder_name:
            self.decoder = UperNetAlignHead2(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish3" == decoder_name:
            self.decoder = UperNetAlignHead3(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish4" == decoder_name:
            self.decoder = UperNetAlignHead4(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish5" == decoder_name:
            self.decoder = down32_DecoderFAM(num_classes, self.body.channels())
        elif "decoderFAMfinish6" == decoder_name:
            self.decoder = UperNetAlignHead5(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv")
        elif "decoderFAMfinish7" == decoder_name:
            self.decoder = UperNetAlignHead4(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=32, conv3x3_type="conv")
        elif "decoderFAMfinish8" == decoder_name:
            self.decoder = UperNetAlignHead6(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderFAMfinish9" == decoder_name:
            self.decoder = UperNetAlignHead7(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderFAMfinish10" == decoder_name:
            self.decoder = UperNetAlignHead8(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderRDD2" == decoder_name: 
            self.decoder = decoder_RDDNet_add()
        elif "decoderRDDwh" == decoder_name:#代码编写有问题
            self.decoder = decoder_RDDNet_wh()
        elif "decoderRDDwh2" == decoder_name:
            self.decoder = decoder_RDDNet_wh2()
        elif "decoderRDDwh3" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat()
        elif "decoderRDDwh4" == decoder_name:
            self.decoder = decoder_RDDNet_psanet()
        elif "decoderRDDwh5" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C384()
        elif "decoderRDDwh6" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024()
        elif "decoderRDDwh7" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_regseg()
        elif "decoderRDDwh8" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1152 ()
        elif "decoderRDDwh9" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_notup ()
        elif "decoderRDD3" == decoder_name:
            self.decoder = decoder_RDDNet_addcat()
        elif "decoderRDDwh10" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1536_channel ()
        elif "decoderRDDwh11" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C2048()
        elif "decoderRDDwh12" == decoder_name:
            self.decoder = decoder_RDDNet_stage4_5_cat_C1024()
        elif "decoderRDDwh13" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C512()
        elif "decoderRDDwh14" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_decoderfinal_attention()
        elif "decoderRDDwh15" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_decoderx32x8_attention()
        elif "decoderRDDwh16" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM()
        elif "decoderRDDwh17" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_channelattention()
        elif "decoderRDDwh18" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S()
        elif "decoderRDDwh19" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_spattention_S()
        elif "decoderRDDwh20" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal()
        elif "decoderRDDwh21" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_attanet()
        elif "decoderRDDwh22" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_spatialattention_S_a()
        elif "decoderRDDwh23" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_attanet_stage45()
        elif "decoderRDDwh24" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_SE_attanet()
        elif "decoderRDDwh25" == decoder_name: #camvid decoder
            self.decoder = decoder_RDDNet_stage5cat_C1024_camvid()
        elif "decoderRDDwh26" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal_ADDs2s4()
        elif "decoderRDDwh27" == decoder_name:
            self.decoder = decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal_camvid()
        elif "decoderRDDwhpaper256" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper()
        elif "decoderRDDwhpaper256PPM" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_PPM()
        elif "decoderRDDwhpaper384" == decoder_name:
            self.decoder = decoder_RDDNet_C384_paper()
        elif "decoderRDDwhpaper640s6" == decoder_name:
            self.decoder = decoder_RDDNet_C640_papers6()
        elif "decoderRDDwhpaper640s6last4stage" == decoder_name:
            self.decoder = decoder_RDDNet_C640_papers6_last4stage()
        elif "decoderRDDwhpaper384s6" == decoder_name:
            self.decoder = decoder_RDDNet_C384_papers6()
        elif "decoderRDDwhpaper640s5" == decoder_name:##final PSA decoder
            self.decoder = decoder_RDDNet_C640_papers5()
        elif "decoderRDDwhpaper640s5camvid" == decoder_name:##final PSA decoder
            self.decoder = decoder_RDDNet_C640_papers5_camvid()
        elif "decoderRDDwhpaperpsaa" == decoder_name:## PSA decoder 论文中的PAS-(a)结构
            self.decoder = decoder_RDDNet_psaa()
        elif "decoderRDDwhpaper640s5normal" == decoder_name:
            self.decoder = decoder_RDDNet_C640_papers5_normal() #不使用selfattention  情况下的decoder
        elif "decoderRDDwhpaper640s6cat" == decoder_name:
            self.decoder = decoder_RDDNet_C640_papers6cat()
        elif "decoderRDDwhpaper896s6" == decoder_name:
            self.decoder = decoder_RDDNet_C896_papers6()
        elif "decoderRDDwhpaper896s6" == decoder_name:
            self.decoder = decoder_RDDNet_C896_papers6()
        elif "decoderRDDwhpaper512" == decoder_name:
            self.decoder = decoder_RDDNet_C512_paper()
        elif "decoderRDDwhpaper768" == decoder_name:
            self.decoder = decoder_RDDNet_C768_paper()
        elif "decoderRDDwhpaper1024" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper()
        elif "decoderRDDwhpaper1536" == decoder_name:
            self.decoder = decoder_RDDNet_C1536_paper()
        elif "decoderRDDwhpaper1024dw" == decoder_name:
            self.decoder = decoder_RDDNet_C1024DW_paper()
        elif "decoderRDDwhpaper1792" == decoder_name:
            self.decoder = decoder_RDDNet_C1792_paper()
        elif "decoderRDDwhpaper256channel5" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_channel5()
        elif "decoderRDDwhpaper256channel3" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_channel3()
        elif "decoderRDDwhpaper256channel1" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_channel1()
        elif "decoderRDDwhpaper256channel2" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_channel2()
        elif "decoderRDDwhpaper1024uafmsp" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten()
        elif "decoderRDDwhpaper1024uafmch" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMChatten()
        elif "decoderRDDwhpaper1024uafmspcat" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_cat()
        elif "decoderRDDwhpaper1024uafmchcat" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMChatten_cat()
        elif "decoderRDDwhpaper1024uafmspcatmean" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_catmean()
        elif "decoderRDDwhpaper1024uafmspchcat" == decoder_name:  #_____________________final decoder
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat()
        elif "decoderRDDwhpaper1024uafmspchcatchannel1" == decoder_name:  #_____________________final decoder-channel1-(32,64,64,128,256)
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel1()
        elif "decoderRDDwhpaper1024uafmspchcatchannel2" == decoder_name:  #_____________________final decoder-channel1-(32,64,128,128,256)
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel2()
        elif "decoderRDDwhpaper1024uafmspchcatchannel4" == decoder_name:  #_____________________final decoder-channel1-(32,64,128,256,512)
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel4()
        elif "decoderRDDwhpaper512uafmspchcat" == decoder_name:
            self.decoder = decoder_RDDNet_C512_paper_UAFMSpatten_Channel_cat()
        elif "decoderRDDwhpaper256uafmspchcat" == decoder_name:
            self.decoder = decoder_RDDNet_C256_paper_UAFMSpatten_Channel_cat()
        elif "decoderRDDwhpaper1024uafmspch" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel()
        elif "decoderRDDwhpaper1280uafmspch" == decoder_name:
            self.decoder = decoder_RDDNet_C1280_paper_UAFMSpatten_Channel()
        elif "decoderRDDwhpaper2560uafmspch" == decoder_name:
            self.decoder = decoder_RDDNet_C2560_paper_UAFMSpatten_Channel()
        elif "decoderRDDwhpaper1024uafmspchcatx16x32" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_x16x32()
        elif "decoderRDDwhpaper1024uafmspchcatx16" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_x16()
        elif "decoderRDDwhpaper1024uafmspchcat345" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_345()
        elif "decoderRDDwhpaper1536uafmspchcat" == decoder_name:
            self.decoder = decoder_RDDNet_C1536_paper_UAFMSpatten_Channel_cat()
        elif "decoderRDDwhpapercamvid1024uafmspchcat" == decoder_name:
            self.decoder = decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_catcamvid()
        elif "decoderRDD4" == decoder_name:
            self.decoder = decoder_RDDNet_catadd()
        elif "decoderRDD5" == decoder_name: 
            self.decoder = decoder_RDDNet_add3x3()
        elif "decoderRDD6" == decoder_name: 
            self.decoder = decoder_RDDNet_add64()
        elif "decoderRDD7" == decoder_name: 
            self.decoder = decoder_RDDNet_add32()
        elif "decoderRDD8" == decoder_name:   #  RAFNet decoder
            self.decoder = decoder_RDDNet_addcommon()
        elif "decoderRDD8selfattention" == decoder_name:   #  RAFNet decoder
            self.decoder = decoder_RDDNet_addcommon_selfattention()
        elif "decoderRDD9" == decoder_name: 
            self.decoder = decoder_RDDNet_addcommon1x1()
        elif "decoderRDD10" == decoder_name: 
            self.decoder = decoder_RDDNet_addFAM()
        elif "decoder29" == decoder_name:
            self.decoder=Exp2_Decoder29(num_classes,self.body.channels())
        elif "decoder26" == decoder_name:
            self.decoder=Exp2_Decoder26(num_classes,self.body.channels())
        elif "decoder26selfattention" == decoder_name:
            self.decoder=Exp2_Decoder26_selfattention(num_classes,self.body.channels())
        elif "decoderSTDCwh" == decoder_name:
            self.decoder = STDC_docker_wh()
        elif "decoderRegSegwh" == decoder_name:
            self.decoder = RegSeg_decoder_wh()
        elif "decoderMFAF" == decoder_name:  #_____________________
            self.decoder = decoder_RDDNet_C1024_paper_MFAF()
        elif "decoderICA" == decoder_name:  #_____________________
            self.decoder = decoder_RDDNet_C1024_paper_ICA()
        else:
            raise NotImplementedError()
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
    def forward(self,x):
        input_shape=x.shape[-2:]
        x=self.stem(x)
        x=self.body(x)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
