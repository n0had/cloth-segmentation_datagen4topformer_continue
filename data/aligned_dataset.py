from data.base_dataset import BaseDataset, Rescale_fixed, Normalize_image
from data.image_folder import make_dataset, make_dataset_test

import os
import cv2
import json
import itertools
import collections
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.image_dir = opt.image_folder
        self.df_path = opt.df_path
        self.width = opt.fine_width
        self.height = opt.fine_height

        # for rgb imgs

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        #transforms_list += [Normalize_image(opt.mean, opt.std)]#############################################
        self.transform_rgb = transforms.Compose(transforms_list)

        self.df = pd.read_csv(self.df_path)
        self.image_info = collections.defaultdict(dict)
        self.df["CategoryId"] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = (
            self.df.groupby("ImageId")["EncodedPixels", "CategoryId"]
            .agg(lambda x: list(x))
            .reset_index()
        )
        size_df = self.df.groupby("ImageId")["Height", "Width"].mean().reset_index()
        temp_df = temp_df.merge(size_df, on="ImageId", how="left")
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row["ImageId"]
            image_path = os.path.join(self.image_dir, image_id)
            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]["orig_height"] = row["Height"]
            self.image_info[index]["orig_width"] = row["Width"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]

        self.dataset_size = len(self.image_info)

    def __getitem__(self, index):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        mask = np.zeros(
            (len(info["annotations"]), self.width, self.height), dtype=np.uint8
        )
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            sub_mask = self.rle_decode(
                annotation, (info["orig_height"], info["orig_width"])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)

        return image_tensor, target_tensor
    
    def debug_saveAllImagePairs(self, savePath):
        indexsaveVal = 1
        indexsavetrain = 1
        for curind in range(400): #range(self.dataset_size):
            if (curind%100 == 99):
                print(curind+1)
            if ((curind%123 == 3) and (indexsaveVal<3)):
                isVal = True
                indexsave = indexsaveVal
                indexsaveVal = indexsaveVal+1
            else:
                isVal=False
                indexsave = indexsavetrain
                indexsavetrain = indexsavetrain+1
            self.debug_saveImagePair(curind, savePath, isVal, indexsave)
    
    def debug_saveImagePair(self, index, savePath, isVal, indexsave):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        orig_width = round(info["orig_height"])
        orig_height = round(info["orig_width"])
        
        mask = np.zeros((len(info["annotations"]), orig_width, orig_height), dtype=np.uint8)
        
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            #sub_mask = self.rle_decode(annotation, (info["orig_height"], info["orig_width"]))
            sub_mask = self.rle_decode(annotation, (info["orig_height"],info["orig_width"]))###################
            
            sub_mask = Image.fromarray(sub_mask)
            #sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BICUBIC)
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), orig_width, orig_height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((orig_width, orig_height), dtype=np.uint8)
        first_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        second_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        third_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        print(type(final_label))
    
    def saveAllImagePairs(self, savePath):
        indexsaveVal = 1
        indexsavetrain = 1
        for curind in range(10002): #range(self.dataset_size):
            if (curind%200 == 199):
                print(curind+1)
            if ((curind%123 == 3) and (indexsaveVal<3)):
                isVal = True
                indexsave = indexsaveVal
                indexsaveVal = indexsaveVal+1
            else:
                isVal=False
                indexsave = indexsavetrain
                indexsavetrain = indexsavetrain+1
            self.saveImagePair(curind, savePath, isVal, indexsave)
    
    #def saveImagePair(self, index, saveImagePath, saveAnnotationPath):
    def saveImagePair(self, index, savePath, isVal, indexsave):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        orig_width = round(info["orig_height"])
        orig_height = round(info["orig_width"])
        
        mask = np.zeros((len(info["annotations"]), orig_width, orig_height), dtype=np.uint8)
        
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            #sub_mask = self.rle_decode(annotation, (info["orig_height"], info["orig_width"]))
            sub_mask = self.rle_decode(annotation, (info["orig_height"],info["orig_width"]))###################
            
            sub_mask = Image.fromarray(sub_mask)
            #sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BICUBIC)
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), orig_width, orig_height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((orig_width, orig_height), dtype=np.uint8)
        first_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        second_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        third_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)
        
        if isVal:
            mid_str = "val_"
            savePath_image = os.path.join(savePath, "images", "validation")
            savePath_annotation = os.path.join(savePath, "annotations", "validation")
        else:
            mid_str = "train_"
            savePath_image = os.path.join(savePath, "images", "training") #savePath +
            savePath_annotation = os.path.join(savePath, "annotations", "training")
        #img_name = img_path
        image_name = "iMaterialist_"+mid_str+str(indexsave).zfill(8)
        
        target_arr = target_tensor.cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_arr) , 'L') #/4? ######################################################## *85? #round(target_arr)?
        
        #output_image
        img.save(os.path.join(savePath_image, image_name+'.jpg'))
        target_img.save(os.path.join(savePath_annotation, image_name+'.png'))#output_annotation.save(os.path.join(savePath_annotation, image_name+'.png'))
        #output_image.save(os.path.join(result_dir, image_name[:-4]+'_generated.png'))
    
    def saveImagePair_strange(self, index, savePath, isVal):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        #img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        orig_height = round(info["orig_height"])
        orig_width = round(info["orig_width"])
        
        mask = np.zeros((len(info["annotations"]), orig_width, orig_height), dtype=np.uint8)
        
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            #sub_mask = self.rle_decode(annotation, (info["orig_height"], info["orig_width"]))
            sub_mask = self.rle_decode(annotation, (info["orig_width"],info["orig_height"]))###################
            
            sub_mask = Image.fromarray(sub_mask)
            #sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BICUBIC)
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), orig_width, orig_height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((orig_width, orig_height), dtype=np.uint8)
        first_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        second_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)
        third_channel = np.zeros((orig_width, orig_height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)
        
        if isVal:
            mid_str = "val_"
            savePath_image = os.path.join(savePath, "images", "validation")
            savePath_annotation = os.path.join(savePath, "annotations", "validation")
        else:
            mid_str = "train_"
            savePath_image = os.path.join(savePath, "images", "training") #savePath +
            savePath_annotation = os.path.join(savePath, "annotations", "training")
        #img_name = img_path
        image_name = "iMaterialist_"+mid_str+str(index).zfill(8)
        
        target_arr = target_tensor.cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_arr/3 * 255) , 'L') #/4? ########################################################
        
        #output_image
        img.save(os.path.join(savePath_image, image_name+'.jpg'))
        target_img.save(os.path.join(savePath_annotation, image_name+'.png'))#output_annotation.save(os.path.join(savePath_annotation, image_name+'.png'))
        #output_image.save(os.path.join(result_dir, image_name[:-4]+'_generated.png'))
        
    #def saveImagePair(self, index, saveImagePath, saveAnnotationPath):
    def saveImagePair_resized(self, index, savePath, isVal):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        mask = np.zeros(
            (len(info["annotations"]), self.width, self.height), dtype=np.uint8
        )
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            sub_mask = self.rle_decode(
                annotation, (info["orig_height"], info["orig_width"])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)
        
        if isVal:
            mid_str = "val_"
            savePath_image = os.path.join(savePath, "images", "validation")
            savePath_annotation = os.path.join(savePath, "annotations", "validation")
        else:
            mid_str = "train_"
            savePath_image = os.path.join(savePath, "images", "training") #savePath +
            savePath_annotation = os.path.join(savePath, "annotations", "training")
        #img_name = img_path
        image_name = "iMaterialist_"+mid_str+str(index).zfill(8)
        
        target_arr = target_tensor.cpu().numpy()
        target_img = Image.fromarray(np.uint8(target_arr/3 * 255) , 'L') #/4? ########################################################
        
        #output_image
        img.save(os.path.join(savePath_image, image_name+'.jpg'))
        target_img.save(os.path.join(savePath_annotation, image_name+'.png'))#output_annotation.save(os.path.join(savePath_annotation, image_name+'.png'))
        #output_image.save(os.path.join(result_dir, image_name[:-4]+'_generated.png'))

        #return image_tensor, target_tensor
    def printTargetTensorDimensions(self, index):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        mask = np.zeros(
            (len(info["annotations"]), self.width, self.height), dtype=np.uint8
        )
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            sub_mask = self.rle_decode(
                annotation, (info["orig_height"], info["orig_width"])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)
        
        print("\n")
        print("\n image tensor")
        print(list(image_tensor.size()))
        print("\n")
        print("\n target tensor")
        print(list(target_tensor.size()))
        print("\n")
        print("\n")
        print("\n target max")
        output_arr = target_tensor.cpu().numpy()
        output_vec=output_arr.reshape(-1)
        print(np.amax(output_vec))
        print("\n target min")
        print(np.amin(output_vec))
        
    def __len__(self):
        return len(self.image_info)

    def name(self):
        return "AlignedDataset"

    def rle_decode(self, mask_rle, shape):
        """
        mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
        shape: (height,width) of array to return
        Returns numpy array according to the shape, 1 - mask, 0 - background
        """
        shape = (round(shape[1]), round(shape[0]))
        
        '''
        string0 = f'{shape[0]:.5f}'
        string1 = f'{shape[1]:.5f}'
        print("\n\n")
        print("string0="+string0+", string1="+string1)
        print("\n\n")
        '''
        
        s = mask_rle.split()
        # gets starts & lengths 1d arrays
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        # gets ends 1d array
        ends = starts + lengths
        # creates blank mask image 1d array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        # sets mark pixles
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # reshape as a 2d mask image
        return img.reshape(shape).T  # Needed to align to RLE direction
