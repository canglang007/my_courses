import cv2
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

class LabelProcessor:

    def __init__(self):

        self.colormap = [
            [0, 0, 0], # background
            [255, 255, 255] # liver
        ]

        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    def encode_label_pix(colormap):
        cm2lb = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lb[(cm[0]*256 + cm[1]) * 256 + cm[2]] = i

        return cm2lb

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        label = np.array(self.color2label[idx], dtype='int64')

        return label

p = LabelProcessor()

if __name__ == '__main__':
    
    total_dice = 0
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0

    name = 'pspnet'
    pred_list = sorted(glob.glob('./'+ name + '/*/*.png'))
    gt_list = sorted(glob.glob('../datasets/chaos_custom/test/Ground/*/*.png'))
    
    for i, (pred1, target1) in tqdm(enumerate(zip(pred_list, gt_list))):
        imgPredict = Image.open(pred1).convert("RGB")
        imgLabel = Image.open(target1).convert("RGB")
        newsize = (512, 512)
        
        imgPredict = imgPredict.resize(newsize, Image.BILINEAR)
        imgLabel = imgLabel.resize(newsize, Image.BILINEAR)
        
        pred_map = p.encode_label_img(imgPredict)
        gt_map = p.encode_label_img(imgLabel)
        
        # Calculate the True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((pred_map == 1) & (gt_map == 1))
        fp = np.sum((pred_map == 1) & (gt_map == 0))
        tn = np.sum((pred_map == 0) & (gt_map == 0))
        fn = np.sum((pred_map == 0) & (gt_map == 1))

        # Calculate the Dice Similarity Coefficient (DSC)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
        total_dice += dice
        
        # Calculate Sensitivity (also called Recall)
        sensitivity = tp / (tp + fn + 1e-10)
        total_sensitivity += sensitivity
        
        # Calculate Specificity
        specificity = tn / (tn + fp + 1e-10)
        total_specificity += specificity
        
        # Calculate Precision
        precision = tp / (tp + fp + 1e-10)
        total_precision += precision
        
    # Calculate the average evaluation metrics over all images
    n_images = len(gt_list)
    avg_dice = total_dice / n_images * 100
    avg_sensitivity = total_sensitivity / n_images * 100
    avg_specificity = total_specificity / n_images * 100
    avg_precision = total_precision / n_images * 100

    print("Dice Coefficient", round(avg_dice, 2),
          "Sensitivity", round(avg_sensitivity, 2),
          "Specificity", round(avg_specificity, 2),
          "Precision", round(avg_precision, 2))