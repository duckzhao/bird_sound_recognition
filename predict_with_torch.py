from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

# 设置超参数
momentum = 0.9
batch_size = 16
class_num = 100
epochs = 100
lr = 0.001
use_gpu = True
bet_name = 'efficientnet-b3'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_filename = 'models/torch_model_first/checkpoint.pth'

# 数据预处理,此次并没有过多的使用反转、偏移等 预处理操作---因此训练集、测试集也无需区分
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 构建dataset
dataset_test = datasets.ImageFolder(root='test_split_img_new', transform=transform)
# print(dataset_test.class_to_idx)
# print(dataset_test.imgs)
# [('test_split_img_new/test3083.wav/6_test3083.png', 3083)]

# 构建dataloader
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
model_ft = EfficientNet.from_name('efficientnet-b3')
# 提取model的全连接层前一步的输入dense维度
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)

# 恢复保存点
checkpoint = torch.load(checkpoint_filename)

# 查看gpu状态
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model_ft.to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 数据预处理,此次并没有过多的使用反转、偏移等 预处理操作---因此训练集、测试集也无需区分
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

model.to(device)

test_answer_label = []
test_answer_prob = []
filenames = [i.replace('test_split_img_new/', '') for i, j in dataset_test.imgs]

for inputs, labels in test_loader:
    # 将输入转到gpu中
    inputs = inputs.to(device)
    outputs = model(inputs)
    prob, preds = torch.max(outputs, 1)
    test_answer_prob.extend(list(prob.data.cpu().numpy()))
    test_answer_label.extend(list(preds.data.cpu().numpy()))

class_indices = {'B000': 0, 'B001': 1, 'B002': 2, 'B003': 3, 'B004': 4, 'B005': 5, 'B006': 6, 'B007': 7, 'B008': 8,
                 'B009': 9, 'B010': 10, 'B011': 11, 'B012': 12, 'B013': 13, 'B014': 14, 'B015': 15, 'B016': 16,
                 'B017': 17, 'B018': 18, 'B019': 19, 'B020': 20, 'B021': 21, 'B022': 22, 'B023': 23, 'B024': 24,
                 'B025': 25, 'B026': 26, 'B027': 27, 'B028': 28, 'B029': 29, 'B030': 30, 'B031': 31, 'B032': 32,
                 'B033': 33, 'B034': 34, 'B035': 35, 'B036': 36, 'B037': 37, 'B038': 38, 'B039': 39, 'B040': 40,
                 'B041': 41, 'B042': 42, 'B043': 43, 'B044': 44, 'B045': 45, 'B046': 46, 'B047': 47, 'B048': 48,
                 'B049': 49, 'B050': 50, 'B051': 51, 'B052': 52, 'B053': 53, 'B054': 54, 'B055': 55, 'B056': 56,
                 'B057': 57, 'B058': 58, 'B059': 59, 'B060': 60, 'B061': 61, 'B062': 62, 'B063': 63, 'B064': 64,
                 'B065': 65, 'B066': 66, 'B067': 67, 'B068': 68, 'B069': 69, 'B070': 70, 'B071': 71, 'B072': 72,
                 'B073': 73, 'B074': 74, 'B075': 75, 'B076': 76, 'B077': 77, 'B078': 78, 'B079': 79, 'B080': 80,
                 'B081': 81, 'B082': 82, 'B083': 83, 'B084': 84, 'B085': 85, 'B086': 86, 'B087': 87, 'B088': 88,
                 'B089': 89, 'B090': 90, 'B091': 91, 'B092': 92, 'B093': 93, 'B094': 94, 'B095': 95, 'B096': 96,
                 'B097': 97, 'B098': 98, 'B099': 99}

account_dict = {}
probility_dict = {}
for p, pro, file_name in zip(test_answer_label, test_answer_prob, filenames):
    test_wav_name = file_name[:12]
    # 获取已经投票的数量
    temp_vote_list = account_dict.get(test_wav_name)
    temp_prob_list = probility_dict.get(test_wav_name)
    # 如果为空，则给字典中创建该元素
    if not temp_vote_list:
        account_dict[test_wav_name] = [list(class_indices.keys())[p]]
        probility_dict[test_wav_name] = [pro]
    else:
        temp_vote_list.append(list(class_indices.keys())[p])
        temp_prob_list.append(pro)

# print(account_dict)
# print(probility_dict)
# 统计完每张图片的数量以后，从value中选出出现频率最高的元素
answer_list = []
answer_list_prob = []
for test_wav_name in account_dict:
    account_list = account_dict[test_wav_name]
    maxlabel = max(account_list, key=account_list.count)
    answer_list.append(test_wav_name+' '+maxlabel)
    # 开始找预测概率最大的prob
    probility_list = probility_dict[test_wav_name]
    max_index = probility_list.index(max(probility_list, key=abs))
    max_pro_label = account_list[max_index]
    answer_list_prob.append(test_wav_name+' '+max_pro_label)

df = pd.Series(answer_list)
df.to_csv('torch_submit_10s_model_EfficientNetB3_10s_guiyihua_maxvote.csv', index=False, header=False)

df = pd.Series(answer_list_prob)
df.to_csv('torch_submit_10s_model_EfficientNetB3_10s_guiyihua_maxprob.csv', index=False, header=False)