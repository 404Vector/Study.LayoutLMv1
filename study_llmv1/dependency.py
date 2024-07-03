import os
from typing import List, Tuple
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import pytesseract
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D
from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification, AdamW

import typer
