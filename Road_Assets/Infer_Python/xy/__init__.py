# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 下午4:55
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : __init__.py
# ------❤❤❤------ #


from .models import My_detection, Build_TRT_model, Build_Ort_model, My_LoggerConfig, Result, str2bool
from .tracker.byte_tracker import BYTETracker

logger = My_LoggerConfig().get_logger()
__all__ = (
    "My_detection",
    "Build_TRT_model",
    "Build_Ort_model",
    "BYTETracker",
    "logger",
    "Result",
    "str2bool"
)
