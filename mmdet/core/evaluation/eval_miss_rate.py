"""
Author: Yuan Yuan.
Date:2019/02/11.
Description: this file defines a function to start Matlab engine on background
             and perform miss rate evaluation.dbEval() is a srcipt for evaluating
             miss rate used in pedestrian detection.the related path should be set
             in Matlab path variable.
"""

import matlab.engine


def eval_miss_rate():
    eng = matlab.engine.start_matlab()
    eng.dbEval(nargout=0)
