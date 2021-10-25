import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f


class CDIPQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CDIPQA, self).__init__(dirPath,split,config,images)

        self.cache_resized = False
        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one

        self.reuse_factor = config['reuse_factor'] if 'reuse_factor' in config else 1.0
        self.calls = 0

        assert images is None

        csv_path = os.path.join(dirPath,'download_urls.csv')
        with open(csv_path) as f:
            download_urls = csv.reader(f)
            self.download_urls = {name:url for name,url in download_urls}

        #check if any are downloaded:
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = f.readlines()
            tar_name1,downloaded1,untared1,calls1 = log[0].split(',')
            ...

            if untared1 and (not untared2 or calls1<calls2):
                #start using tar_name1
            elif untarted2:
                #start using tar_name2:

        if none ready:
            #how to stall?


            if 'overfit' in config and config['overfit']:
                splitFile = 'overfit_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                readFile = json.loads(f.read())
                if split in readFile:
                    subdirs = readFile[split]
                    new_subdirs=[]
                    for sub in subdirs:
                        if '.' in sub:
                            new_subdirs.append(sub)
                        else:
                            for a in 'abcdefghijklmnopqrstuvwxyz':
                                new_subdirs.append(sub+'.'+a)
                    toUse=[]
                    for subdir in new_subdirs:
                        try:
                            with open(os.path.join(dirPath,subdir+'.list')) as lst:
                                toUse += [path.strip() for path in lst.readlines()]
                        except FileNotFoundError:
                            print('{} not found'.format(os.path.join(dirPath,subdir+'.list')))
                    imagesAndAnn = []
                    for path in toUse:#['images']:
                        try:
                            name = path[path.rindex('/')+1:]
                        except ValueError:
                            name = path
                        imagesAndAnn.append( (name,os.path.join(dirPath,path+'.png'),os.path.join(dirPath,path+'.layout.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit(1)
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                #if os.path.exists(jsonPath):
                #    org_path = imagePath
                #    if self.cache_resized:
                #        path = os.path.join(self.cache_path,imageName+'.png')
                #    else:
                #        path = org_path

                #    rescale=1.0
                #    if self.cache_resized:
                #        rescale = self.rescale_range[1]
                #        if not os.path.exists(path):
                #            org_img = img_f.imread(org_path)
                #            if org_img is None:
                #                print('WARNING, could not read {}'.format(org_img))
                #                continue
                #            resized = img_f.resize(org_img,(0,0),
                #                    fx=self.rescale_range[1], 
                #                    fy=self.rescale_range[1], 
                #                    )
                #            img_f.imwrite(path,resized)
                rescale=1.0
                path = imagePath
                self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale })
                #else:
                #    print('{} does not exist'.format(jsonPath))
                #    print('No json found for {}'.format(imagePath))
                #    #exit(1)
        self.errors=[]




    def parseAnn(self,ocr,s):
        image_h=ocr['height']
        image_w=ocr['width']
        ocr=ocr['blocks']

        block_score_sum=0
        line_count=0
        for block in ocr:
            t,l,b,r = block['box']
            h=b-t
            w=r-l
            if w==0 or h==0:
                continue
            squareness = min(0.4,h/w)
            area_whole = h*w
            area_covered = 0 #we'll assume lines don't overlap
            num_lines=0
            for para in block['paragraphs']:
                for line in para['lines']:
                    num_lines+=1
                    for word in line['words']:
                        top,left,bottom,right = word['box']
                        height = bottom-top
                        width = right-left
                        area_covered+=height*width
            if num_lines>1:
                area_score = area_covered/area_whole
            else:
                area_score = 0
            total_score = area_score+squareness
            block_score_sum += total_score*num_lines
            line_count += num_lines
        block_score = block_score_sum/line_count if line_count>0 else 0
        use_blocks = block_score>self.block_score_thresh
        #print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, qa

