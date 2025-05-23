import numpy as np
import SimpleITK as sitk
import torch
import shutil
import math
import copy
import pandas as pd

# from sklearnex import patch_sklearn
# patch_sklearn()

def data_selection_wLabel(args, df):
    date_standard = 'treatedate'
    # date_standard = 'initialdate'
    survival_type = args.survival_type
    year = args.year
    
    df['label'] = 3
    if survival_type == 'OS':
        duration = abs(pd.to_datetime(df['lastdate']) - pd.to_datetime(df[date_standard])).dt.days
        df.loc[(duration >= year * 365) & (df['dead'] == 0), 'label'] = 0 # negative
        
        df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 1), 'label'] = 1 # positive
        df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 2), 'label'] = 2 # excluded
        df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['hospital'] == 'EUMC'), 'label'] = 1 # positive for EUMC
        
        # if df['hospital'][idx] == 'EUMC':
        #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1), 'label'] = 1 # positive
        # else:
        #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 1), 'label'] = 1 # positive
        #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 2), 'label'] = 2 # excluded
    elif survival_type == 'RFS':
        duration = abs(pd.to_datetime(df['lastdate']) - pd.to_datetime(df[date_standard])).dt.days
        df.loc[(duration >= year * 365) & (df['relapse'] == 1), 'label'] = 0 # negative
        df.loc[(duration < year * 365) & (duration > 0) & (df['relapse'] != 1), 'label'] = 1 # positive
    
    df_new = df.loc[df['label'].isin([0, 1])]
    return df_new

def data_selection(df):
    df1 = df[df['classification cancer'].isin([1,2])]
    
    df1 = df1[df1['cancerimaging'].isin([1,2,3,4,'1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
    df1 = df1[df1['cancerimagingT'].isin([1,2,3,4,'1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
    df1 = df1[df1['cancerimagingN'].isin([0,1,2,3,4,'0','1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
    df2 = df1[df1['cancerimagingM'].isin([0,1,'0','1','1a','1b','1c'])]
    
    df2 = df2.loc[df2['sex'].isin(['M', 'F'])]
    df2 = df2.loc[df2['sm'].isin(['N', 'Y'])]
    df3 = df2.loc[df2['locationcancer'].isin([1,2,3,4,5])]

    return df3


def slice_preprocessing_with_metadata(metadata):
    img = metadata.pixel_array.astype(np.float32)

    if ('RescaleSlope' in metadata) and ('RescaleIntercept' in metadata):
        img = (img * metadata.RescaleSlope) + metadata.RescaleIntercept
    # img = (img / 2**metadata.BitsStored)

    # if('WindowCenter' in metadata):
    #     if(type(metadata.WindowCenter) == pydicom.multival.MultiValue):
    #         window_center = float(metadata.WindowCenter[0])
    #         window_width = float(metadata.WindowWidth[0])
    #         lwin = window_center - (window_width / 2.0)
    #         rwin = window_center + (window_width / 2.0)
    #     else:
    #         window_center = float(metadata.WindowCenter)
    #         window_width = float(metadata.WindowWidth)
    #         lwin = window_center - (window_width / 2.0)
    #         rwin = window_center + (window_width / 2.0)
    # else:
    #     lwin = np.min(img)
    #     rwin = np.max(img)

    # img[np.where(img < lwin)] = lwin
    # img[np.where(img > rwin)] = rwin
    # img = img - lwin

    if(metadata.PhotometricInterpretation == 'MONOCHROME1'):
        # img[np.where(img < lwin)] = lwin
        # img[np.where(img > rwin)] = rwin
        # img = img - lwin
        img = 2**metadata.BitsStored - img

    return img


def resample_with_spacing(args, img, metadata):
    # img : (z,x,y)
    # img_sitk : (x,y,z)
    img_sitk = convert_to_sitk(img, metadata)

    orig_size = img_sitk.GetSize()
    orig_spacing = img_sitk.GetSpacing()
    out_spacing = args.spacing
    out_size = [int(orig_size[0] * (orig_spacing[0]/out_spacing[0])),
                int(orig_size[1] * (orig_spacing[1]/out_spacing[1])),
                int(orig_size[2] * (orig_spacing[2]/out_spacing[2]))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img_sitk.GetPixelIDValue())
    img_sitk_resampled = resample.Execute(img_sitk)

    # img_resampled = sitk.GetArrayFromImage(img_sitk_resampled)
    return img_sitk_resampled


def convert_to_sitk(img, metadata):
    sitk_volume = sitk.GetImageFromArray(img)
    sitk_volume.SetSpacing([metadata.PixelSpacing[0], metadata.PixelSpacing[1], metadata.SliceThickness])
    return sitk_volume


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# def calculate_accuracy(outputs, targets):
#     # output : [B,2]
#     # target : [B,]
#     with torch.no_grad():
#         batch_size = targets.size(0)

#         _, pred = outputs.topk(1, 1, largest=True, sorted=True)
#         pred = pred.t()
#         correct = pred.eq(targets.view(1, -1))
#         # n_correct_elems = correct.float().sum().item()
#         n_correct_elems = correct.float().sum()

#         return n_correct_elems / batch_size

def calculate_accuracy(outputs, targets):
    # output : [B,2] ex) [[0.4, 0.6]]
    # target : [B,2] ex) [[0.0, 1.0]]
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(torch.argmax(targets, dim=-1))
        # n_correct_elems = correct.float().sum().item()
        n_correct_elems = correct.float().sum()

        return n_correct_elems / batch_size

# def calculate_accuracy(outputs, targets):
#     # output : [B,1]
#     # target : [B,1]
#     with torch.no_grad():
#         batch_size = targets.size(0)
        
#         pred = torch.zeros_like(outputs)
#         pred[outputs>=0.5] = 1
#         pred[outputs< 0.5] = 0
#         correct = pred.eq(targets)
#         n_correct_elems = correct.float().sum()
    
#     return n_correct_elems / batch_size


# def calculate_auc(outputs, targets):
    


def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth.tar"):
    torch.save(state, save_dir + '/' + filename)
    if is_best:
        shutil.copyfile(save_dir + '/' + filename, save_dir + "/checkpoint_best.pth.tar")


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class ProgressMeter_wID:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, ID, batch):
        entries = [self.prefix + "[{:>9s}]".format(ID) + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.n_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


#%%
# import clip

class CLIPloss_v1(torch.nn.Module):
    def __init__(self, args):
        super(CLIPloss_v1, self).__init__()
        
        self.args = args
        model, _ = clip.load("ViT-B/32")
        self.model = model.cuda(self.args.gpu)
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.clinical_info = self.args.clinical_features
        
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.args.gpu)
        
    def forward(self, output, input_CI):
        # output : [b, 512]
        # input_CI : [b, 1, # of Info]
        feature_by_CLIP = torch.zeros((output.shape[0],len(self.clinical_info),512)).to(output.device) #[b, # of Info, 512]
        
        for b in range(output.shape[0]):
            input_text = []
            for i in range(len(self.clinical_info)):
                input_text.append(f"a lung cancer patient photo of {self.clinical_info[i]} {input_CI[b,0,i]}")
        
            tokenized_text = clip.tokenize(input_text) #[9, 77]
            
            with torch.no_grad():
                text_feature = self.model.encode_text(tokenized_text.cuda(self.args.gpu)) #[9, 512]
            
            feature_by_CLIP[b,:,:] = text_feature
        
        logits = torch.matmul(output.unsqueeze(0), feature_by_CLIP.permute(1,2,0)) #[1, b, 512] x [# of Info, 512, b] = [# of Info, b, b]
        labels = torch.eye(output.shape[0]).to(output.device)
        labels = labels.unsqueeze(0).repeat(logits.shape[0],1,1)
        
        loss = self.criterion(logits, labels)
        
        return loss
    