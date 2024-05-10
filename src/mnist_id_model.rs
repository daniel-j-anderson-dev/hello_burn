pub mod config;
pub mod dataset_batch;

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Debug, Module)]
pub struct MnistIdModel<B: Backend> {
    convolution_layer_1: Conv2d<B>,
    convolution_layer_2: Conv2d<B>,
    adaptive_average_pooling: AdaptiveAvgPool2d,
    dropout_regularizer: Dropout,
    linear_layer_1: Linear<B>,
    linear_layer_2: Linear<B>,
    activation_function: Relu,
}
impl<B: Backend> MnistIdModel<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let output = images.reshape([batch_size, 1, height, width]);

        let output = self.convolution_layer_1.forward(output);
        let output = self.dropout_regularizer.forward(output);
        let output = self.convolution_layer_2.forward(output);
        let output = self.dropout_regularizer.forward(output);
        
        let output = self.adaptive_average_pooling.forward(output);
        let output = output.reshape([batch_size, 16 * 8 * 8]);
        let output = self.linear_layer_1.forward(output);
        let output = self.dropout_regularizer.forward(output);
        let output = self.activation_function.forward(output);
        
        let output = self.linear_layer_2.forward(output);
        
        return output;
    }
}
