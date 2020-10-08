from model.oversegment_loss import build_oversegmented_targets_multiscale
from model.overseg_box_detector import build_box_predictions
from utils.bb_merging import TextLine
import torch
import utils.img_f as img_f
import numpy as np
import math,random
from utils.forms_annotations import calcCorners
def calcPoints(x,y,r,h,w):
    rx = x + math.cos(r)*w
    ry = y - math.sin(r)*w
    lx = x - math.cos(r)*w
    ly = y + math.sin(r)*w

    tx = x - math.sin(r)*h
    ty = y - math.cos(r)*h
    bx = x + math.sin(r)*h
    by = y + math.cos(r)*h

    return lx,ly,rx,ry,tx,ty,bx,by

def drawPoly(img,pts,color,thck=1):
    for i in range(-1,len(pts)-1):
        img_f.line(img,(int(pts[i][0]),int(pts[i][1])),(int(pts[i+1][0]),int(pts[i+1][1])),color,thck)

num_classes=2
numBBTypes=2
numAnchors=4
numBBParams=6
H=300
W=700
#scale = [ (16,16), (32,32), (64,64) ]
scale = [ (32,32), (64,64) ]
grid_sizesH=[H//s[0] for s in scale]
grid_sizesW=[W//s[0] for s in scale]
pred_boxes = [torch.zeros(1,1,1,1,1)]*3
pred_cls = [torch.zeros(1,1,1,1,1)]*3
pred_conf = [torch.zeros(1,1,1,1)]*3

targs=[]
#varying sizes
#yb=100
#t=0
#for h in range(40,41,4):
#    w = h*3
#    r = math.pi/10#-math.pi*(7/10)
#    y = yb
#    for x in range(150,300,300):
#        lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
#        targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])
#        t+=1
#        y+=1
#        r += math.pi/10
#        if r>math.pi:
#            r-=math.pi*2
#        #r = (random.random()*math.pi/2)-math.pi/4
#    yb+=1.1*(h+w)

boxes = [
        [450,110,-math.pi*3/10,40,120],
        #[150,110,-math.pi*7/10,40,120],
        #[150,110,-math.pi*1/10,40,120],
        [299,105,math.pi/20,40,40],
        [345,80,-math.pi/40,40,80],
        #[140,100,0,40,120]
        ]
t=len(boxes)
for x,y,r,h,w in boxes:
    lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
    targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])

#Totally random
#t=50
#for i in range(t):
#    w = random.random()*1000+8
#    h = random.random()*50+8
#    x = random.random()*(H-200) +100
#    y = random.random()*(W-200) +100
#    #r = (random.random()*math.pi*2)-math.pi
#    sig = math.pi/24
#    if random.random()>0.2:
#        if random.random()>0.5:
#            r = random.gauss(0,sig)
#        else:
#            r = random.gauss(math.pi,sig)
#    else:
#        if random.random()>0.5:
#            r = random.gauss(math.pi/2,sig)
#        else:
#            r = random.gauss(-math.pi/2,sig)
#    if r>math.pi:
#        r-=math.pi*2
#    elif r<-math.pi:
#        r+=math.pi*2
#    lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
#    targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])

target_sizes= [t]
target= torch.FloatTensor(1,t,13+1)
for t,targ in enumerate(targs):
    target[0,t,:13]=torch.FloatTensor(targ)

nGT, masks, conf_masks, t_Ls, t_Ts, t_Rs, t_Bs, t_rs, tconf_scales, tcls_scales, pred_covered, gt_covered, recall, precision, pred_covered_noclass, gt_covered_noclass, recall_noclass, precision_noclass = build_oversegmented_targets_multiscale(pred_boxes, pred_conf, pred_cls, target, target_sizes, num_classes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')


# Handle target variables
t_Ls = [t.type(torch.FloatTensor) for t in t_Ls]
t_Ts = [t.type(torch.FloatTensor) for t in t_Ts]
t_Rs = [t.type(torch.FloatTensor) for t in t_Rs]
t_Bs = [t.type(torch.FloatTensor) for t in t_Bs]
t_rs = [t.type(torch.FloatTensor) for t in t_rs]
tconf_scales = [t.type(torch.FloatTensor) for t in tconf_scales]
tcls_scales = [t.type(torch.FloatTensor) for t in tcls_scales]

ys = []
for level in range(len(t_Ls)):
    level_y = torch.cat([ torch.stack([tconf_scales[level],t_Ls[level],t_Ts[level],t_Rs[level], t_Bs[level],t_rs[level]],dim=2), tcls_scales[level].permute(0,1,4,2,3)], dim=2)
    ys.append(level_y.view(level_y.size(0),level_y.size(1)*level_y.size(2),level_y.size(3),level_y.size(4)))

    display_y = level_y[0].permute(2,3,0,1).contiguous().view(-1,level_y.size(2))

    #print('level_y: {}'.format(display_y[display_y[:,0]>0.5]))
gt_boxes = build_box_predictions(ys,scale,ys[0].device,numAnchors,numBBParams,numBBTypes)
assert((gt_boxes[0,gt_boxes[0,:,0]>0.5,2] < gt_boxes[0,gt_boxes[0,:,0]>0.5,4]).all())
#print('gt: {}'.format(gt_boxes[0,gt_boxes[0,:,0]>0.5]))

img = np.zeros([H,W,3])
for t in targs:
    corners = calcCorners(*t[0:5])
    #img_f.polylines(img,np.array([[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]],np.int32).reshape((-1,1,2)),True,(0,255,0),3)
    #drawPoly(img,[[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]],(0,255,0),3)
    #print([[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]])
    drawPoly(img,corners,(0,255,0),3)


all_textlines=[]
for bb in gt_boxes[0]:
    if bb[0]>0.5:
        #all_textlines.append(TextLine(torch.FloatTensor([1,t[5],t[10],t[7],t[12],0,0,1])))
        #img_f.rectangle(img,(t[5],t[10]),(t[7],t[12]),(0,0,255),1)
        all_textlines.append(TextLine(torch.FloatTensor(bb)))
        #img_f.rectangle(img,bb[1:3].numpy(),bb[3:5].numpy(),(0,0,255))
        #img_f.line(img,(round(bb[1].item()),round(bb[2].item())),(round(bb[3].item()),round(bb[4].item())),(0,0,255))
        img_f.rectangle(img,(round(bb[1].item()),round(bb[2].item())),(round(bb[3].item()),round(bb[4].item())),(0,0,255),2)
        print(bb)

first_textline=all_textlines[0]

for tl in all_textlines[1:]:
    first_textline.merge(tl)

#img_f.polylines(img,np.array(first_textline.polyPoints(),np.int32).reshape((-1,1,2)),True,(255,0,0),1)
drawPoly(img,first_textline.polyPoints(),(255,0,0),1)
#for p in first_textline.polyPoints():
#    img[int(p[1]),int(p[0]),1]=255
for t,b in first_textline.pairPoints():
    color = [100+random.randint(0,154),100+random.randint(0,154),100+random.randint(0,154)]
    img[int(t[1]-1):int(t[1]+1),int(t[0]-1):int(t[0]+1),:]=color
    img[int(b[1]-1):int(b[1]+1),int(b[0]-1):int(b[0]+1),:]=color
for p in first_textline._top_points:
    img[int(p[1]-2):int(p[1]+2),int(p[0]-2):int(p[0]+2),:]=[0,1,1]
    img[int(p[1]-1):int(p[1]+1),int(p[0]-1):int(p[0]+1),:]=[1,0,1]
for p in first_textline._bot_points:
    img[int(p[1]-2):int(p[1]+2),int(p[0]-2):int(p[0]+2),:]=[1,0,1]
    img[int(p[1]-1):int(p[1]+1),int(p[0]-1):int(p[0]+1),:]=[1,1,0]
print('final poly: {}'.format(first_textline.polyPoints()))

img_f.imshow('x',img)
img_f.show()


#from model.optimize import optimizeRelationships
#import torch
#relPred=torch.tensor([.5,.4,.6,.2,.3,.1,.1])
#relNodes=[(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)]
#gtNodeN=[2,1,1,1,2]
#res=optimizeRelationships(relPred,relNodes,gtNodeN)
#import pdb;pdb.set_trace()
#print(res)
#

#import numpy as np
#import matplotlib.pyplot as plt
#
#iterations=100000
#
#warmup_steps=10000
#def lrf(step_num):
#    return min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
#
#cycle_size=5000
#decay_rate=0.99995
#min_lr_mul=0.0001
#low_lr_mul=0.3
#def decayCycle (step_num):
#                cycle_num = step_num//cycle_size
#                decay = decay_rate**step_num
#                if cycle_num%2==0: #even, rising
#                    return decay*((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
#                else: #odd
#                    return -decay*(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1) + 1-(1-min_lr_mul)*(1-decay)
#
#iters_in_trailoff = iterations-(2*cycle_size)
#def oneCycle (step_num):
#                cycle_num = step_num//cycle_size
#                if step_num<cycle_size: #rising
#                    return ((1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + low_lr_mul
#                elif step_num<cycle_size*2: #falling
#                    return (1-(1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
#                else: #trail off
#                    t_step_num = step_num-(2*cycle_size)
#                    return low_lr_mul*(iters_in_trailoff-t_step_num)/iters_in_trailoff + min_lr_mul*t_step_num/iters_in_trailoff
#
#x = [i for i in range(0,iterations)]
##y = [lrf(i) for i in x]
##y = [decayCycle(i) for i in x]
#y = [oneCycle(i) for i in x]
#y=np.array(y)
#
#print('max val: {}'.format(y.max()))
#print('mean val: {}'.format(y.mean()))
#plt.plot(x,y,'.')
#plt.show()
