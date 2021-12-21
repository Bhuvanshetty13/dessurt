import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random
from utils.crop_transform import CropBoxTransform
from utils import augmentation
from collections import defaultdict, OrderedDict
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT
import timeit

import utils.img_f as img_f


def collate(batch):
    if any(b['mask_label'] is not None for b in batch):
        mask_labels = []
        mask_labels_batch_mask = torch.FloatTensor(len(batch))
        for bi,b in enumerate(batch):
            if b['mask_label'] is None:
                mask_labels_batch_mask[bi]=00
                mask_labels.append( torch.FloatTensor(1,1,b['img'].shape[2],b['img'].shape[3]).fill_(0))
            else:
                mask_labels_batch_mask[bi]=1
                mask_labels.append( b['mask_label'] )
        mask_labels = torch.cat(mask_labels,dim=0)
    else:
        mask_labels = None
        mask_labels_batch_mask = None

    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'bb_gt': [b.get('bb_gt') for b in batch], 
            'imgName': [b.get('imgName') for b in batch],
            'scale': [b.get('scale') for b in batch],
            'cropPoint': [b.get('cropPoint') for b in batch],
            'transcription': [b.get('transcription') for b in batch],
            'metadata': [b.get('metadata') for b in batch],
            'form_metadata': [b.get('form_metadata') for b in batch],
            'questions': [b.get('questions') for b in batch],
            'answers': [b.get('answers') for b in batch],
            'mask_label': mask_labels,
            'mask_labels_batch_mask': mask_labels_batch_mask,
            #'mask_label': torch.cat([b['mask_label'] for b in batch],dim=0) if batch[0]['mask_label'] is not None else [b['mask_label'] for b in batch],
            'pre-recognition': [b.get('pre-recognition') for b in batch],
            "bart_logits": torch.cat([b['bart_logits'] for b in batch],dim=0) if 'bart_logits' in batch[0] else None,
            "bart_last_hidden": torch.cat([b['bart_last_hidden'] for b in batch],dim=0) if 'bart_last_hidden' in batch[0] else None,
            }

def getMask(shape,boxes):
    mask = torch.FloatTensor(1,1,shape[2],shape[3]).fill_(0)
    for box in boxes:
        #tlX,tlY,trX,trY,brX,brY,blX,blY = box[0:8]
        if isinstance(box,list):
            box = np.array(box)
        points = box[0:8].reshape(4,2)
        #mask[0,0,round(t):round(b+1),round(l):round(r+1)]=1 
        #img_f.fillConvexPoly(img,((tlX,tlY),(trX,trY),(brX,brY),(blX,blY)),1)
        img_f.fillConvexPoly(mask[0,0],points,1)
    return mask


class QADataset(torch.utils.data.Dataset):
    """
    Class for reading dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.train = split=='train'
        self.questions = config['questions']
        self.max_qa_len_in = config['max_qa_len_in'] if 'max_qa_len_in' in config else None
        self.max_qa_len_out = config['max_qa_len_out'] if 'max_qa_len_out' in config else None
        if self.max_qa_len_in is None and self.max_qa_len_out is None and 'max_qa_len' in config:
            self.max_qa_len_in = config['max_qa_len']
            self.max_qa_len_out = config['max_qa_len']

        self.cased = config.get('cased',False)

        self.color = config['color'] if 'color' in config else False
        self.rotate = config['rotation'] if 'rotation' in config else False
        #patchSize=config['patch_size']
        if 'crop_params' in config and config['crop_params'] is not None:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
        self.rescale_to_crop_size_first = config['rescale_to_crop_size_first'] if 'rescale_to_crop_size_first' in config else False
        self.rescale_to_crop_width_first = config['rescale_to_crop_width_first'] if 'rescale_to_crop_width_first' in config else False
        if self.rescale_to_crop_size_first or self.rescale_to_crop_width_first:
            self.crop_size = config['crop_params']['crop_size']
        if type(self.rescale_range) is float:
            self.rescale_range = [self.rescale_range,self.rescale_range]
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                if self.rescale_to_crop_size_first:
                    self.cache_path = os.path.join(dirPath,'cache_match{}x{}'.format(*config['crop_params']['crop_size']))
                elif self.rescale_to_crop_width_first:
                    self.cache_path = os.path.join(dirPath,'cache_matchHx{}'.format(config['crop_params']['crop_size'][1]))
                else:
                    self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else False
        self.aug_params = config['additional_aug_params'] if 'additional_aug_params' in config else {}


        self.do_masks=False    

        self.ocr_out_dim = 97 #EasyOCR
        #self.char_to_ocr = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž"
        self.char_to_ocr = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_ocr = {char:i+1 for i,char in enumerate(self.char_to_ocr)} #+1 as 0 is the blank token
        self.one_hot_conf = 0.9

        #t#self.opt_history = defaultdict(list)#t#

        self.crop_to_data = False




    def __len__(self):
        return len(self.images)


    def qaAdd(self,qa,question,answer,bb_ids=None,in_bbs=[],out_bbs=None,mask_bbs=[]):
        if all([pair['question']!=question for pair in qa]): #prevent duplicate q
            qa.append({
                'question':question,
                'answer':answer,
                'bb_ids':bb_ids,
                'in_bbs':in_bbs,
                'out_bbs':out_bbs,
                'mask_bbs':mask_bbs
                })

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        #t#ticFull=timeit.default_timer()#t#
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index]['imageName']
        annotationPath = self.images[index]['annotationPath']
        #print(annotationPath)
        rescaled = self.images[index]['rescaled']
        if type(annotationPath) is int:
            annotations = annotationPath
        elif annotationPath.endswith('.json'):
            try: 
                with open(annotationPath) as annFile:
                    annotations = json.loads(annFile.read())
            except FileNotFoundError:
                print("ERROR, could not open "+annotationPath)
                return self.__getitem__((index+1)%self.__len__())
            except json.decoder.JSONDecodeError as e:
                print(e)
                print('Error reading '+annotationPath)
                return self.__getitem__((index+1)%self.__len__())
        else:
            annotations=annotationPath

        #Load image
        #t#tic=timeit.default_timer()#t#
        if imagePath is not None:
            try:
                np_img = img_f.imread(imagePath, 1 if self.color else 0)#*255.0
            except FileNotFoundError as e:
                print(e)
                print('ERROR, could not find: '+imagePath)
                return self.__getitem__((index+1)%self.__len__())
            if np_img is None or np_img.shape[0]==0:
                print("ERROR, could not open "+imagePath)
                return self.__getitem__((index+1)%self.__len__())
            if np_img.max()<=1:
                np_img*=255
        else:
            np_img = None#np.zeros([1000,1000])

        if self.crop_to_data:
            #This exists for the IAM dataset so we don't include the form prompt text (which would be easy to cheat from)
            x1,y1,x2,y2 = self.getCrop(annotations)
            np_img = np_img[y1:y2,x1:x2]


        #
        if scaleP is None:
            s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        else:
            s = scaleP

        if self.rescale_to_crop_size_first:
            if rescaled!=1:
                raise NotImplementedError('havent implemented caching with match resizing')
        
            scale_height = self.crop_size[0]/np_img.shape[0]
            scale_width = self.crop_size[1]/np_img.shape[1]
            #if np_img.shape[0] > np_img.shape[1]:
            #    #portiat oriented document
            #else:
            #    #landscape oriented document
            #    #We will switch between scaleing to fit and scaling to roughly 
            #    #the size of a portiat document (will get cropped severly).
            #    if random.random()<0.5:
            #        #fit
            #        scale_height = self.crop_size[0]/np_img.shape[0]
            #        scale_width = self.crop_size[1]/np_img.shape[1]
            #    else:
            #        #match
            #        scale_height = self.crop_size[0]/np_img.shape[1]
            #        scale_width = self.crop_size[1]/np_img.shape[0]

            scale = min(scale_height, scale_width)
            partial_rescale = s*scale
            s=partial_rescale
        elif self.rescale_to_crop_width_first:
            if rescaled!=1:
                raise NotImplementedError('havent implemented caching with match resizing')
            scale = self.crop_size[1]/np_img.shape[1]
            partial_rescale = s*scale
            s=partial_rescale
        else:
            partial_rescale = s/rescaled
        
        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['image read and setup'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        

        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))

        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['parseAnn'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        #Parse annotation file
        bbs,ids,trans, metadata, form_metadata, questions_and_answers = self.parseAnn(annotations,s)


        if not self.train:
            questions_and_answers = self.images[index]['qa']
            #But the scale doesn't match! So fix it
            for qa in questions_and_answers:
                for bb_name in ['in_bbs','out_bbs','mask_bbs']:
                    if qa[bb_name] is not None:
                        qa[bb_name] = [ [s*v for v in bb] for bb in qa[bb_name] ]



        if np_img is None:
            np_img=metadata['image']
            del metadata['image']
        if partial_rescale!=1:
            np_img = img_f.resize(np_img,(0,0),
                    fx=partial_rescale,
                    fy=partial_rescale,
            )


        if len(np_img.shape)==2:
            np_img=np_img[...,None] #add 'color' channel
        if self.color and np_img.shape[2]==1:
            np_img = np.repeat(np_img,3,axis=2)
        
        #set up for cropping
        outmasks=False
        if self.do_masks:
            assert self.questions==1 #right now, we only allow 1 qa pair if using masking
            mask_bbs=[]
            mask_ids=[]
            for i,qa in enumerate(questions_and_answers):
                #if len(qa)==5:
                #q,a,bb_ids,inmask_bbs,outmask_bbs,blank_bbs = qa
                inmask_bbs = qa['in_bbs']
                outmask_bbs = qa['out_bbs']
                blank_bbs = qa['mask_bbs']
                if outmask_bbs is not None:
                    outmasks=True
                    mask_bbs+=inmask_bbs+outmask_bbs+blank_bbs
                    mask_ids+=  ['in{}_{}'.format(i,ii) for ii in range(len(inmask_bbs))] + \
                                ['out{}_{}'.format(i,ii) for ii in range(len(outmask_bbs))] + \
                                ['blank{}_{}'.format(i,ii) for ii in range(len(blank_bbs))]
                else:
                    mask_bbs+=inmask_bbs+blank_bbs
                    mask_ids+=  ['in{}_{}'.format(i,ii) for ii in range(len(inmask_bbs))] + \
                                ['blank{}_{}'.format(i,ii) for ii in range(len(blank_bbs))]
                #mask_ids+=(['in{}'.format(i)]*len(inmask_bbs)) + (['out{}'.format(i)]*len(outmask_bbs)) + (['blank{}'.format(i)]*len(blank_bbs))
            if 'pre-recognition_bbs' in form_metadata:
                mask_bbs+= form_metadata['pre-recognition_bbs']
                mask_ids+=  ['recog{}'.format(ii) for ii in range(len(form_metadata['pre-recognition_bbs']))]
            mask_bbs = np.array(mask_bbs)
            #if len(mask_bbs.shape)==1:
            #    mask_bbs=mask_bbs[None]

        #Do crop
        if self.transform is not None:
            if 'word_boxes' in form_metadata:
                raise NotImplementedError('have not added mask_bbs')
                word_bbs = form_metadata['word_boxes']
                dif_f = bbs.shape[2]-word_bbs.shape[1]
                blank = np.zeros([word_bbs.shape[0],dif_f])
                prep_word_bbs = np.concatenate([word_bbs,blank],axis=1)[None,...]
                crop_bbs = np.concatenate([bbs,prep_word_bbs],axis=1)
                crop_ids=ids+['word{}'.format(i) for i in range(word_bbs.shape[0])]
            elif self.do_masks and len(mask_bbs.shape)==2:
                if bbs.shape[0]>0 and mask_bbs.shape[0]>0:
                    crop_bbs = np.concatenate([bbs,mask_bbs])
                elif mask_bbs.shape[0]>0:
                    crop_bbs = mask_bbs
                else:
                    crop_bbs = bbs
                #print(crop_bbs)
                crop_ids = ids+mask_ids
            else:
                crop_bbs = bbs
                crop_ids = ids
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": crop_bbs[None,...],
                'bb_auxs':crop_ids,
                #'word_bbs':form_metadata['word_boxes'] if 'word_boxes' in form_metadata else None
                #"line_gt": {
                #    "start_of_line": start_of_line,
                #    "end_of_line": end_of_line
                #    },
                #"point_gt": {
                #        "table_points": table_points
                #        },
                #"pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']


            new_q_inboxes=defaultdict(list)
            if outmasks:
                new_q_outboxes=defaultdict(list)
            else:
                new_q_outboxes=None
            new_q_blankboxes=defaultdict(list)
            new_recog_boxes={}
            if 'word_boxes' in form_metadata:
                saw_word=False
                word_index=-1
                for i,ii in enumerate(out['bb_auxs']):
                    if not saw_word:
                        if type(ii) is str and 'word' in ii:
                            saw_word=True
                            word_index=i
                    else:
                        assert 'word' in ii
                bbs = out['bb_gt'][0,:word_index]
                ids= out['bb_auxs'][:word_index]
                form_metadata['word_boxes'] = out['bb_gt'][0,word_index:,:8]
                word_ids=out['bb_auxs'][word_index:]
                form_metadata['word_trans'] = [form_metadata['word_trans'][int(id[4:])] for id in word_ids]
            elif self.do_masks:
                orig_idx=0
                for ii,(bb_id,bb) in enumerate(zip(out['bb_auxs'],out['bb_gt'][0])):
                    if type(bb_id) is int:
                        assert orig_idx==ii
                        orig_idx+=1
                    elif bb_id.startswith('in'):
                        nums = bb_id[2:].split('_')
                        i=int(nums[0])
                        new_q_inboxes[i].append(bb)
                    elif bb_id.startswith('out'):
                        nums = bb_id[3:].split('_')
                        i=int(nums[0])
                        new_q_outboxes[i].append(bb)
                    elif bb_id.startswith('blank'):
                        nums = bb_id[5:].split('_')
                        i=int(nums[0])
                        new_q_blankboxes[i].append(bb)
                    elif bb_id.startswith('recog'):
                        i=int(bb_id[5:])
                        new_recog_boxes[i]=bb
                bbs = out['bb_gt'][0,:orig_idx]
                ids= out['bb_auxs'][:orig_idx]

                #Put boxes back in questions_and_answers
                for i in range(len(questions_and_answers)):
                    questions_and_answers[i]['in_bbs'] = new_q_inboxes[i]
                    if outmasks:
                        questions_and_answers[i]['out_bbs'] = new_q_outboxes[i]
                    questions_and_answers[i]['mask_bbs'] = new_q_blankboxes[i]
            else:
                bbs = out['bb_gt'][0]
                ids= out['bb_auxs']

            if questions_and_answers is not None:
                questions=[]
                answers=[]
                #questions_and_answers = [(q,a,qids) for q,a,qids in questions_and_answers if qids is None or all((i in ids) for i in qids)]
                questions_and_answers = [qa for qa in questions_and_answers if qa['bb_ids'] is None or all((i in ids) for i in qa['bb_ids'])]

        if questions_and_answers is not None:
            if len(questions_and_answers) > self.questions:
                questions_and_answers = random.sample(questions_and_answers,k=self.questions)
            if len(questions_and_answers)==0:
                print('Had no questions...')
                return self.getitem((index+1)%len(self))
            new_q_inboxes= [qa['in_bbs'] for qa in questions_and_answers]
            new_q_outboxes= [qa['out_bbs'] for qa in questions_and_answers]
            new_q_blankboxes= [qa['mask_bbs'] for qa in questions_and_answers]
            if self.cased:
                questions = [qa['question'] for qa in questions_and_answers]
                answers = [qa['answer'] for qa in questions_and_answers]
            else:
                questions = [qa['question'].lower() for qa in questions_and_answers]
                answers = [qa['answer'].lower() for qa in questions_and_answers]
        else:
            questions=answers=None




            ##tic=timeit.default_timer()
        if self.augment_shade and self.augment_shade>random.random():
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))

        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0

        if self.do_masks:
            assert len(new_q_inboxes)<=1
            assert new_q_outboxes is None or len(new_q_outboxes)<=1
            mask = getMask(img.shape,new_q_inboxes[0])
            img = torch.cat((img,mask),dim=1)
            for blank_box in new_q_blankboxes[0]:
                assert(img.shape[1]==2)
                x1,y1,x2,y2,x3,y3,x4,y4 = blank_box[:8]
                img_f.polylines(img[0,0],np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]),True,0) #blank on image
                img_f.polylines(img[0,-1],np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]),True,-1) #flip mask to indicate it was blanked
                #img[0,:-1,t:b+1,l:r+1]=0 #blank on image
                #img[0,-1,t:b+1,l:r+1]=-1 #flip mask to indicate was blanked
            #img = addMask(img,new_q_outboxes[0])
            if outmasks:
                mask_label = getMask(img.shape,new_q_outboxes[0])
            else:
                mask_label = None
            #import pdb;pdb.set_trace()
        else:
            mask_label = None
        #if pixel_gt is not None:
        #    pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
        #    pixel_gt = torch.from_numpy(pixel_gt)


        if bbs is not None:
            bbs = convertBBs(bbs[None,...],self.rotate,0)
            if bbs is not None:
                bbs=bbs[0]
            else:
                bbs = torch.FloatTensor(1,0,5+8+1)
        else:
            bbs = torch.FloatTensor(1,0,5+8+1)
        #if 'word_boxes' in form_metadata:
        #     form_metadata['word_boxes'] = convertBBs(form_metadata['word_boxes'][None,...],self.rotate,0)[0,...]

        #import pdb;pdb.set_trace()
        if trans is not None:
            transcription = [trans[id] for id in ids]
        else:
            transcription = None
        if 'pre-recognition' in form_metadata:
            #format similar to output of EasyOCR
            pre_recog=[]
            for i,bb in new_recog_boxes.items():
                string = form_metadata['pre-recognition'][i]
                #char_prob = torch.FloatTensor(len(string),self.ocr_out_dim).fill_((1-self.one_hot_conf)/(self.ocr_out_dim-1)) #fill with 5% distributed to all non-char places
                #for pos,char in enumerate(string):
                #    char_prob[pos,self.char_to_ocr[char]]=self.one_hot_conf
                char_prob = [self.char_to_ocr[char] for char in string if char in self.char_to_ocr]
                pre_recog.append( (bb[0:8].reshape(4,2),(string,char_prob),None) )
        else:
            pre_recog = None

        #t#time = timeit.default_timer()#t#
        #t#self.opt_history['remainder'].append(time-tic)#t#
        #t#self.opt_history['Full get_item'].append(time-ticFull)#t#
        #t#self.print_opt_times()#t#

        return {
                "img": img,
                "bb_gt": bbs,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "transcription": transcription,
                "metadata": [metadata[id] for id in ids if id in metadata],
                "form_metadata": None,
                "questions": questions,
                "answers": answers,
                "mask_label": mask_label,
                "pre-recognition": pre_recog
                }


    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time data {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
