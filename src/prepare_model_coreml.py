import os
import sys

import numpy as np
import torch

import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torchvision import models
import coremltools as ct

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, values):

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(self.tanh(self.W1(query)))

        # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
        # min_score = tf.reshape(min_score, [-1, 1, 1])
        # score_mask = tf.greater_equal(score, min_score)
        # score_mask = tf.cast(score_mask, tf.float32)
        # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

        # attention_weights shape == (batch_size, max_length, 1)
        score = self.sigmoid(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)

        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output


class GA_Net(nn.Module):
    def __init__(self):
        super(GA_Net, self).__init__()

        cnn = models.efficientnet_b0()
        cnn.classifier = Identity()

        self.TimeDistributed = TimeDistributed(cnn)
        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)

    def forward(self, x):

        x = self.TimeDistributed(x)

        x_v = self.WV(x)

        x_a, x_s = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x, w_a

class GA_Net_features(nn.Module):
    def __init__(self, cnn):
        super(GA_Net_features, self).__init__()

        self.TimeDistributed = TimeDistributed(cnn)

    def forward(self, x):

        x = self.TimeDistributed(x)

        return x

class GA_Net_attn_output(nn.Module):
    def __init__(self):
        super(GA_Net_attn_output, self).__init__()

        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)


    def forward(self, x):

        x_v = self.WV(x)
        x_a, x_s = self.Attention(x, x_v)
        x = self.Prediction(x_a)

        return x, x_s
    
class GA_Net_score(nn.Module):
    def __init__(self):
        super(GA_Net_score, self).__init__()

        self.feat = None
        self.WV = None
        self.W1 = None
        self.V = None
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.Prediction = None


    def forward(self, x):

        x_f = self.feat(x)
        x_v = self.WV(x_f)
        
        x_s = self.V(self.tanh(self.W1(x_f)))
        x_s = self.sigmoid(x_s)

        x = self.Prediction(x_v)

        return x, x_s


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
    def forward(self, x):
        # x = x.repeat(1, 1, 1, 3)
        x = x.permute(0,2,3,1)
        return torch.div(torch.sub(torch.div(x, 255.0), self.mean), self.std).permute((0, 3, 1, 2))


class GA_Net_features_norm(nn.Module):
    def __init__(self, cnn):
        super(GA_Net_features_norm, self).__init__()
        self.norm = NormLayer()
        self.cnn = cnn

    def forward(self, x):
        x = x.reshape((1,3,256,256))
        x = self.norm(x)
        x = self.cnn(x)
        return x


class ResampleImage(nn.Module):
    def __init__(self, target_spacing=(0.75, 0.75)):
        super(ResampleImage, self).__init__()
        self.target_spacing = target_spacing

    def forward(self, x, spacing):

        return x

#Model
device = 'cpu'
# model_path = '/work/jprieto/data/remote/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/train_out/model/model_ga_0115.pt'
# model_path = '/work/jprieto/data/remote/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/train_out/model/model_ga_0310.pt'
# model_path = '/work/jprieto/data/remote/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/train_out/model/model_ga_0328.pt'
# model_path = 'train_out/model/model_ga_0328.pt'

model_path = '/Users/juan/UNC/data/us-famli/model_ga_0328.pt'


model = GA_Net()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

model_features = GA_Net_features(model.TimeDistributed.module)
cnn_efficient = model.TimeDistributed.module

cnn_efficient_norm = GA_Net_features_norm(cnn_efficient)

model_attn_output = GA_Net_attn_output()
model_attn_output.WV = model.WV
model_attn_output.Attention = model.Attention
model_attn_output.Prediction = model.Prediction

cnn_efficient.eval()

example_input = torch.rand(1, 3, 256, 256)
cnn_efficient_traced = torch.jit.trace(cnn_efficient, example_input)
out_feat = cnn_efficient_traced(example_input)
# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.

scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)

cnn_efficient_traced_ct = ct.convert(
    cnn_efficient_traced,
    inputs=[image_input]
 )

cnn_efficient_traced_ct.save(model_path.replace('.pt', '_features.mlpackage'))


example_input_pred = torch.rand(1, 1, 1280)
model_attn_output.eval()
model_attn_output_traced = torch.jit.trace(model_attn_output, example_input_pred)
out_pred = model_attn_output_traced(example_input_pred)
# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model_attn_output_traced_ct = ct.convert(
    model_attn_output_traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input_pred.shape)]
 )

model_attn_output_traced_ct.save(model_path.replace('.pt', '_prediction.mlpackage'))



model_score = GA_Net_score()
model_score.feat = model.TimeDistributed.module
model_score.WV = model.WV
model_score.W1 = model.Attention.W1
model_score.V = model.Attention.V
model_score.Prediction = model.Prediction

model_score.eval()

example_input = torch.rand(1, 3, 256, 256)
model_score_traced = torch.jit.trace(model_score, example_input)
out_model_score = model_score_traced(example_input)

scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)

ga_output = ct.TensorType(name="ga")
score_output = ct.TensorType(name="score")

model_score_traced_ct = ct.convert(
    model_score_traced,
    inputs=[image_input],
    outputs=[ga_output, score_output]
 )

model_score_traced_ct.save(model_path.replace('.pt', '_score.mlpackage'))