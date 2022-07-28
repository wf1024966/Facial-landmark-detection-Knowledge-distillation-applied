import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, gt_label, student_output, teacher_output, alpha):
        super().__init__()
        self.student_output = student_output
        self.teacher_output = teacher_output
        self.gt_label = gt_label
        self.alpha = alpha
    
    def soft_loss(self, student_output, teacher_output):
        return F.mse_loss(student_output, teacher_output)
    
    def hard_loss(self, student_output, gt_label):
        return F.mse_loss(student_output, gt_label)

    def loss_fn(self, soft_loss, hard_loss):
        return soft_loss * self.alpha + hard_loss * (1 - self.alpha)