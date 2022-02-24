import json
import sys
import os
from change_checkpoint_reset_for_training import readRemoveWrite
from make_run import create


if len(sys.argv)==1:
    print('pretrained-checkpoint newID dirLocation')
    exit()

pretrained_path=sys.argv[1]
#itera=int(sys.argv[2])
new_id=int(sys.argv[2])
new_dir_loc=sys.argv[3]

path = pretrained_path.split('/')
name = path[-2]
file_name = path[-1]
iter_loc = file_name.find('iteration')
if iter_loc == -1:
    itera='latest'
else:
    itera = int(file_name[iter_loc+9:-4])
    if itera<1000000:
        itera = '{}k'.format( round(itera/1000))
    elif itera==1000000:
        itera = '1M'
    else:
        itera = '{:.2}m'.format( itera/1000000)
print('from '+itera)


broken = name.split('_')
old_id = broken[-1]
new_name='naf_'+('_'.join(broken[1:-2]))+'_PTfrom{}i{}_{}'.format(old_id,itera,new_id)
print(new_name)

old_cf = 'configs/cf_{}.json'.format(name)
new_cf = 'configs/cf_{}.json'.format(new_name)

destination = os.path.join(new_dir_loc,new_name)
os.mkdir(destination)
readRemoveWrite(pretrained_path,destination)



with open(old_cf) as f:
    cf = json.load(f)

image_size = cf['model']['image_size']
new_dataset= {
        "data_set_name": "NAFQA",
        "data_dir": "../data/forms",
        "shuffle": True,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "batch_size": 1,
        "num_workers": 6,
        "rescale_range":[0.9,1.1],
        "rescale_to_crop_size_first": True,
        "crop_params": {
              "crop_size":image_size,
              "pad":0,
              "rot_degree_std_dev": 1
              },
          "questions":1,
          "max_qa_len": 9999000,
          "use_json": True
            }
new_val =  {
        "shuffle": False,
        "rescale_range":[1,1],
        "crop_params": {
              "crop_size":image_size,
              "pad":0,
              "random":False
              }
    }



cf['name']=new_name
cf['data_loader']=new_dataset

cf['validation']=new_val

cf['trainer']['iterations']=100000
cf['trainer']['val_step']=5000
cf['trainer']['save_step']=40000
cf['trainer']["save_step_minor"]= 1024 
cf['trainer']["monitor"]= "val_E_json_CE"
cf['trainer']["monitor_mode"]= "min"

#set drop in LR
cf['trainer']["use_learning_schedule"]= "multi_rise then ramp_to_lower"
cf['trainer']["lr_down_start"]= 85000
cf['trainer']["ramp_down_steps"]= 5000
cf['trainer']["lr_mul"]= 0.1



with open(new_cf,'w') as f:
    json.dump(cf,f,indent=4)
print(new_cf)
create(new_name)
