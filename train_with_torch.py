from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
import time
import os
import copy
import tqdm

# 设置超参数
momentum = 0.9
batch_size = 48
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
dataset_train = datasets.ImageFolder(root='train_split_img_new', transform=transform)
dataset_val = datasets.ImageFolder(root='dev_split_img_new', transform=transform)
# 对应文件夹的label
# print(dataset_train.class_to_idx)
# 训练样本数量
dset_sizes = len(dataset_train)
# 验证样本数量
dset_sizes_val = len(dataset_val)
# print("dset_sizes_val Length:", dset_sizes_val)

# 构建dataloader
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
dataloaders = {'train': train_loader, 'valid': val_loader}


# 定义调整学习率的函数
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    # 根据epoch进行的轮次， 通过手动方式调整优化器中的学习率
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('lr is set to: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# 准备model
# 不使用预训练model
model_ft = EfficientNet.from_name('efficientnet-b3')
# 提取model的全连接层前一步的输入dense维度
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)
criterion = nn.CrossEntropyLoss()

# 给model 配置gpu
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()

# 配置优化器, model中所有参数都训练
optimizer = torch.optim.Adam((model_ft.parameters()), lr=lr)

# 定义训练model的函数
def train_model(model_ft, criterion, optimizer, lr_scheduler):
    since = time.time()
    model = model_ft
    # 记录当前model在验证集上最佳的准确率，用于保存最佳model到本地
    best_acc = 0

    # 记录训练中每个epochs的 acc 和 loss 以便观察
    train_acc_history = []
    val_acc_history = []
    train_losses = []
    val_losses = []
    # 可以看为用于记录优化器学习率变化的列表
    LRs = []

    # 如果之前训练过，可以通过判断 checkpoint文件在不在决定是否 续着训练
    if os.path.exists(checkpoint_filename):
        print('load the checkpoint!')
        checkpoint = torch.load(checkpoint_filename)
        model_ft.load_state_dict(checkpoint['state_dict'])

        # 因为此时我们训练了所有层，所以当前 优化器中待训练的参数 和 checkpoint中的 数量是不一致的，此时无法加载优化器，只能重新设置一个优化器
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_acc = checkpoint['best_acc']
        # model.class_to_idx = checkpoint['mapping']

        # 如果不是加载历史的 model的优化器 ，optimizer.param_group[0]['lr']为空会报错
        LRs = [optimizer.param_group[0]['lr']]

        print(best_acc, LRs)

    model_ft.to(device)

    # 预定义一个当前最佳model的内存对象---在训练循环中更替
    best_model_wts = copy.deepcopy(model.state_dict())

    # 开始进入正式训练流程
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        # 在每个epoch训练之前调整学习率
        optimizer = lr_scheduler(optimizer, epoch)
        # 每一个epoch 都包括 train 和 valid 两个过程，指定train或者eval 主要是为了BN和Dropout层 在训练和测试 时候有所不同，需要说明
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 评估

            # 用于统计每个 训练 epoch 中的 loss 和 corrects
            running_loss = 0.0
            runing_corrects = 0.0

            # 开始进入真实的取数据-训练 循环
            # 使用for循环 从 dataloader 中取数据，每次取出指定的一个 batch size
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                # tensor([76, 40, 77, 36, 85, 54, 11, 50, 88, 47, 40, 69, 26, 89, 53, 59, 74, 77,
                #         94, 68, 38, 74, 77, 29, 18,  7, 33, 15, 25, 42, 32, 57])
                # print(labels)

                # 将输入转到gpu中
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()

                # 计算梯度并反向传播---->尽在训练时进行
                # torch.set_grad_enabled(mode) 当 mode=True 时对with下的操作记录梯度，否则不记录梯度，训练时开始记录梯度，验证时=False
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 对 outputs进行如下转换得到preds，才能和 labels的形式对应起来
                    _, preds = torch.max(outputs, 1)
                    # tensor([17, 69, 97, 37,  6, 96, 64, 54, 41, 54, 90, 37,  7, 86, 12, 44, 76, 65,
                    #         96, 30, 15, 89, 72, 40, 12,  3, 99, 78, 96,  7, 96, 47], device='cuda:0'))
                    # print(preds)

                    # 仅在训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 累加每个 batch 的 损失
                running_loss += loss.item() * inputs.size(0)
                runing_corrects += torch.sum(preds == labels.data)

            # 迭代数据的for循环结束，标志着一个epoch训练结束，统计该epoch的平均 loss和acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = runing_corrects / len(dataloaders[phase].dataset)

            # 打印当前epoch的训练时间和准确率，损失的信息
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 每个epoch中 都需要在 valid 验证结束后，根据 valid 的 loss 和 acc 判断当前model是否当前最佳，并保存当前最佳的model
            if phase == 'valid' and epoch_acc > best_acc:
                # 更新最佳的 model 准确率
                best_acc = epoch_acc
                # 更新最佳的 model 训练参数 到 内存中，以便在训练结束后，直接从内存中加载最佳的model，而不用再从 checkpoint文件中去读取
                best_model_wts = copy.deepcopy(model.state_dict())
                # state中保存了完整的当前 model 的 checkpoint，主要用于以后恢复当前训练点，进行继续训练。
                # 如果只想保存当前model，以后用于预测任务，则仅需保存 model.state_dict()/best_model_wts 即可
                state = {
                    'state_dict': best_model_wts,  # model 每层权重参数
                    'best_acc': best_acc,  # 当前验证最佳准确率
                    'optimizer': optimizer.state_dict()  # 当前训练过程中 优化器的参数
                }
                torch.save(state, checkpoint_filename)

            # 记录每个 epoch 中的 train 的 acc 和 loss 变化数值，用于可视化训练信息
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)

        # 当每个 epoch 训练完成后，记录当前epoch 优化器的学习率
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    # 当所有 epochs 训练完成后，打印训练花费的整体时间和 epoch最佳准确率
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 模型训练最后一轮的权重不一定是最佳权重，因此需要手动设置 训练过程中最佳的 权重 到model中
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, val_losses, train_acc_history, train_losses, LRs


train_model(model_ft, criterion, optimizer, exp_lr_scheduler)


