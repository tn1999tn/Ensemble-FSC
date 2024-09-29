import yaml
import os

all_roots = {}
all_roots["ILSVRC"] = "PATH-TO-IMAGENET" #0
all_roots["Omniglot"] = "PATH-TO-omniglot" #1
all_roots["Quick Draw"] = "PATH-TO-quickdraw" #2
all_roots["CUB"] = "PATH-TO-CUB" #3
all_roots["VGG Flower"] = "PATH-TO-vggflower" #4
all_roots["Aircraft"] = "PATH-TO-aircraft"  #5
all_roots["Traffic Signs"] = "PATH-TO-traffic" #6
all_roots["MSCOCO"] = "PATH-TO-coco" #7
all_roots["Textures"] = "PATH-TO-dtd" #8
all_roots["Fungi"] = "PATH-TO-fungi" #9
all_roots["MNIST"] = "PATH-TO-mnist" #10
all_roots["CIFAR10"] = "PATH-TO-cifar10" #11
all_roots["CIFAR100"] = "PATH-TO-cifar100" #12
all_roots["miniImageNet"] = "/data1/jiangzhen/tn/LightningFSL-main/miniImageNet" #13

Data = {}

Data["DATA"] = {}


Data["IS_TRAIN"] = 0

names = list(all_roots.keys())
roots = list(all_roots.values())


Data["DATA"]["TEST"] = {}

Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[13]]
Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[13]]


# 5 way 5 shot example
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
Data["DATA"]["TEST"]["BATCH_SIZE"] = 8



Data["OUTPUT"] = "/ensemble_result"
Data["MODEL"] = {}

Data["MODEL"]["NAME"] = "evaluation-pn-ft"
Data["GPU_ID"] = 2


Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 0

Data["AUG"] = {}

# tieredImageNet and CUB
# Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
# Data["AUG"]["STD"] = [0.229, 0.224, 0.225]

# miniImageNet
Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

Data["DATA"]["IMG_SIZE"] = 84

Data["MODEL"]["BACKBONE"] = 'resnet12'
# Data["MODEL"]["BACKBONE"] = 'resnet50'
# Data["MODEL"]["BACKBONE"] = 'clip'

Data["MODEL"]["PRETRAINED"] = '../pretrained_models/pn_miniImageNet_res12.ckpt'# for example

Data["DATA"]["NUM_WORKERS"] = 8

#miniImageNet
Data["AUG"]["TEST_CROP"] = True

Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 2000

# some examples of gradient-based methods. Hyperparameters need to be tuned by using search_hyperparameter.py
Data["MODEL"]["TYPE"] = "fewshot_finetune"
Data["MODEL"]["CLASSIFIER"] = "finetune"
# Data["MODEL"]["CLASSIFIER"] = "eTT"
Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune_batchsize,query_feedingbatchsize,epoch,backbone_lr,classifer_lr,use_alpha,use_beta, mode
#Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,True,True,"NCC"]# tsa
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,True,"NCC"]# URL
#Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"cc"]# CC
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,"eTT"]# eTT

# other adaptation classifiers
#Data["MODEL"]["TYPE"] = "Episodic_Model"
# Data["MODEL"]["CLASSIFIER"] = "LR"
# Data["MODEL"]["CLASSIFIER"] = "metaopt"
#["MODEL"]["CLASSIFIER"] = "proto_head"
# Data["MODEL"]["CLASSIFIER"] = "MatchingNet"

if not os.path.exists('/configs/mini_evaluation_5/'):
   os.makedirs('/configs/mini_evaluation_5/')
with open('/configs/mini_evaluation_5/ft_res12_pn.yaml', 'w') as f:
    yaml.dump(data=Data, stream=f)
