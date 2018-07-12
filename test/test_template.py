#!/usr/bin/python
# -*- coding: UTF-8 -*-

import unittest
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import cv2

from context import video_obj


class TestTemplate(unittest.TestCase):
    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
