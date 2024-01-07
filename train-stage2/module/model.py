#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : baseline-t5                 #
#                                                                   #
#                   @File Name    : model.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2022/8/12 14:51                 #
#                                                                   #
#                   @Last Update  : 2022/8/12 14:51                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''
from torch import nn
from transformers import MT5ForConditionalGeneration


class MT5PForSequenceClassification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t5p = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model_path)

    def forward(self, input_ids=None, labels=None):
        outputs = self.t5p(input_ids=input_ids, labels=labels)
        decoder_loss = outputs.loss
        loss = decoder_loss
        return loss

    def generate(self, input_tensors,
                 decoder_start_token_id=None,
                 eos_token_id=None,
                 max_length=512):
        text_output = self.t5p.generate(input_tensors,
                                        decoder_start_token_id=decoder_start_token_id,
                                        eos_token_id=eos_token_id,
                                        max_length=max_length).cpu().numpy()
        return text_output
