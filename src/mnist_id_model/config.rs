use super::MnistIdModel;

use burn::{
    config::Config,
    nn::{conv::Conv2dConfig, pool::AdaptiveAvgPool2dConfig, DropoutConfig, LinearConfig, Relu},
    tensor::backend::Backend,
};

#[derive(Debug, Config)]
pub struct MnistIdModelConfig {
    output_class_count: usize,
    hidden_layer_size: usize,
    #[config(default = "0.5")]
    dropout_rate: f64,
}
impl MnistIdModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MnistIdModel<B> {
        MnistIdModel {
            convolution_layer_1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            convolution_layer_2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            adaptive_average_pooling: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation_function: Relu::new(),
            linear_layer_1: LinearConfig::new(16 * 8 * 8, self.hidden_layer_size).init(device),
            linear_layer_2: LinearConfig::new(self.hidden_layer_size, self.output_class_count)
                .init(device),
            dropout_regularizer: DropoutConfig::new(self.dropout_rate).init(),
        }
    }
}
