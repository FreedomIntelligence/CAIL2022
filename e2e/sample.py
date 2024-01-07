#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5-segment                 #
#                                                                   #
#                   @File Name    : sample.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2022/8/14 11:58                 #
#                                                                   #
#                   @Last Update  : 2022/8/14 11:58                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''

class Sample:
    def __init__(self, id, segment):
        self.id = id
        self.segment = segment
        self.text = []
        self.summary = []

    def label(self, text_list, summary_sent_list):
        for i in range(len(text_list)):
            self.text.append({"sentence": text_list[i]["sentence"], "important": text_list[i]["important"]})
            if text_list[i]["important"] != 0:
                self.summary.append(summary_sent_list[text_list[i]["important"] - 1])

    def label_for_test(self, text_sent_list):
        for i in range(len(text_sent_list)):
            self.text.append({"sentence": text_sent_list[i], "important": 0})

    def to_json(self):
        return {
            "id": self.id,
            "segment": self.segment,
            "text": self.text,
            "summary": self.summary
        }
