use candle_core::{Device, Result, Tensor};
use candle_nn::{conv1d, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

use crate::aliasfree::AmpBlock;

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?;
    pos + (neg * negative_slope)?
}

struct KernelPredictor {
    input_conv: Conv1d,
    res_convs: Vec<(Conv1d, Conv1d)>,
    kernel_conv: Conv1d,
    bias_conv: Conv1d,
    conv_in: usize,
    conv_out: usize,
    conv_layers: usize,
    kpnet_conv_size: usize,
}

impl KernelPredictor {
    fn new(
        cond_channels: usize,
        conv_in: usize,
        conv_out: usize,
        conv_layers: usize,
        kpnet_conv_size: usize,
        kpnet_hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg5 = Conv1dConfig {
            padding: 2,
            ..Default::default()
        };
        let input_conv = conv1d(cond_channels, kpnet_hidden, 5, cfg5, vb.pp("input_conv").pp("0"))?;

        let mut res_convs = Vec::with_capacity(3);
        let res_vb = vb.pp("residual_convs");
        for i in 0..3 {
            let cfg3 = Conv1dConfig {
                padding: 1,
                ..Default::default()
            };
            let c1 = conv1d(kpnet_hidden, kpnet_hidden, 3, cfg3, res_vb.pp(format!("{i}.1")))?;
            let c2 = conv1d(kpnet_hidden, kpnet_hidden, 3, cfg3, res_vb.pp(format!("{i}.3")))?;
            res_convs.push((c1, c2));
        }

        let kernel_out_dim = conv_layers * conv_in * conv_out * kpnet_conv_size;
        let bias_out_dim = conv_layers * conv_out;
        let cfg3 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let kernel_conv = conv1d(kpnet_hidden, kernel_out_dim, 3, cfg3, vb.pp("kernel_conv"))?;
        let bias_conv = conv1d(kpnet_hidden, bias_out_dim, 3, cfg3, vb.pp("bias_conv"))?;

        Ok(Self {
            input_conv,
            res_convs,
            kernel_conv,
            bias_conv,
            conv_in,
            conv_out,
            conv_layers,
            kpnet_conv_size,
        })
    }

    fn forward(&self, cond: &Tensor) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut h = leaky_relu(&self.input_conv.forward(cond)?, 0.1)?;
        for (c1, c2) in &self.res_convs {
            let y = leaky_relu(&c1.forward(&h)?, 0.1)?;
            let y = leaky_relu(&c2.forward(&y)?, 0.1)?;
            h = (&h + y)?;
        }
        let kernels_flat = self.kernel_conv.forward(&h)?;
        let biases_flat = self.bias_conv.forward(&h)?;

        let (b, _, t) = kernels_flat.dims3()?;
        let kernels_flat = kernels_flat.reshape((
            b,
            self.conv_layers,
            self.conv_in,
            self.conv_out,
            self.kpnet_conv_size,
            t,
        ))?;
        let biases_flat = biases_flat.reshape((b, self.conv_layers, self.conv_out, t))?;

        let mut kernels = Vec::with_capacity(self.conv_layers);
        let mut biases = Vec::with_capacity(self.conv_layers);
        for l in 0..self.conv_layers {
            kernels.push(kernels_flat.narrow(1, l, 1)?.squeeze(1)?);
            biases.push(biases_flat.narrow(1, l, 1)?.squeeze(1)?);
        }
        Ok((kernels, biases))
    }
}

fn location_variable_convolution(
    x: &Tensor,
    kernel: &Tensor,
    bias: &Tensor,
    dilation: usize,
    hop_size: usize,
) -> Result<Tensor> {
    let (b, c_in, x_len) = x.dims3()?;
    let (_, _, c_out, ksize, n_frames) = kernel.dims5()?;
    let pad = dilation * (ksize - 1) / 2;
    let x = x.pad_with_zeros(2, pad, pad)?;

    let mut frame_outputs = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame_start = f * hop_size;
        let frame_len = hop_size.min(x_len.saturating_sub(frame_start));
        if frame_len == 0 {
            continue;
        }
        let k_f = kernel.narrow(4, f, 1)?.squeeze(4)?;
        let bias_f = bias.narrow(2, f, 1)?;

        let mut sample_outputs = Vec::with_capacity(frame_len);
        for s in 0..frame_len {
            let center = frame_start + s + pad;
            let mut taps = Vec::with_capacity(ksize);
            for k in 0..ksize {
                let offset = (k as i64 - (ksize as i64 - 1) / 2) * dilation as i64;
                let idx = (center as i64 + offset).clamp(0, x.dim(2)? as i64 - 1) as usize;
                taps.push(x.narrow(2, idx, 1)?);
            }
            let window = Tensor::cat(&taps, 2)?.reshape((b, 1, c_in * ksize))?;
            let k_mat = k_f.reshape((b, c_in * ksize, c_out))?;
            let out_s = window.matmul(&k_mat)?.transpose(1, 2)?;
            sample_outputs.push((out_s + &bias_f)?);
        }
        frame_outputs.push(Tensor::cat(&sample_outputs, 2)?);
    }
    Tensor::cat(&frame_outputs, 2)
}

struct LvcBlock {
    convt_pre: ConvTranspose1d,
    amp: AmpBlock,
    conv_blocks: Vec<Conv1d>,
    kernel_predictor: KernelPredictor,
    dilations: Vec<usize>,
    cond_hop_length: usize,
    channels: usize,
}

impl LvcBlock {
    fn new(
        channels: usize,
        cond_channels: usize,
        stride: usize,
        cond_hop_length: usize,
        dilations: &[usize],
        kpnet_conv_size: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: stride / 2 + stride % 2,
            output_padding: stride % 2,
            stride,
            ..Default::default()
        };
        let convt_pre = conv_transpose1d(channels, channels, 2 * stride, cfg, vb.pp("convt_pre").pp("1"))?;
        let amp = AmpBlock::new(channels, &[1, 3, 5], vb.pp("amp_block"), device)?;

        let mut conv_blocks = Vec::with_capacity(dilations.len());
        let cb_vb = vb.pp("conv_blocks");
        for (i, &d) in dilations.iter().enumerate() {
            let cfg = Conv1dConfig {
                padding: d,
                dilation: d,
                ..Default::default()
            };
            conv_blocks.push(conv1d(channels, channels, 3, cfg, cb_vb.pp(i).pp("1"))?);
        }

        let kernel_predictor = KernelPredictor::new(
            cond_channels,
            channels,
            channels * 2,
            dilations.len(),
            kpnet_conv_size,
            64,
            vb.pp("kernel_predictor"),
        )?;

        Ok(Self {
            convt_pre,
            amp,
            conv_blocks,
            kernel_predictor,
            dilations: dilations.to_vec(),
            cond_hop_length,
            channels,
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let x = self.convt_pre.forward(&leaky_relu(x, 0.2)?)?;
        let mut h = self.amp.forward(&x)?;

        let (kernels, biases) = self.kernel_predictor.forward(cond)?;
        for i in 0..self.dilations.len() {
            let y = leaky_relu(&h, 0.2)?;
            let y = leaky_relu(&self.conv_blocks[i].forward(&y)?, 0.2)?;
            let lvc_out =
                location_variable_convolution(&y, &kernels[i], &biases[i], 1, self.cond_hop_length)?;
            let lvc_out = lvc_out.narrow(1, 0, self.channels * 2)?;
            let gate = lvc_out.narrow(1, 0, self.channels)?;
            let filt = lvc_out.narrow(1, self.channels, self.channels)?;
            let gated = (candle_nn::ops::sigmoid(&gate)? * filt.tanh()?)?;
            let min_len = h.dim(2)?.min(gated.dim(2)?);
            h = (h.narrow(2, 0, min_len)? + gated.narrow(2, 0, min_len)?)?;
        }
        Ok(h)
    }
}

pub struct UnivNet {
    conv_pre: Conv1d,
    blocks: Vec<LvcBlock>,
    conv_post: Conv1d,
}

impl UnivNet {
    pub fn new(
        cond_channels: usize,
        noise_dim: usize,
        channels: usize,
        strides: &[usize],
        dilations: &[usize],
        kpnet_conv_size: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let cfg7 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv_pre = conv1d(noise_dim, channels, 7, cfg7, vb.pp("conv_pre"))?;

        let mut blocks = Vec::with_capacity(strides.len());
        let blk_vb = vb.pp("blocks");
        let mut cumulative_hop = 1usize;
        for (i, &s) in strides.iter().enumerate() {
            cumulative_hop *= s;
            blocks.push(LvcBlock::new(
                channels,
                cond_channels,
                s,
                cumulative_hop,
                dilations,
                kpnet_conv_size,
                blk_vb.pp(i),
                device,
            )?);
        }

        let conv_post = conv1d(channels, 1, 7, cfg7, vb.pp("conv_post").pp("1"))?;

        Ok(Self {
            conv_pre,
            blocks,
            conv_post,
        })
    }

    pub fn forward(&self, noise: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_pre.forward(noise)?;
        for block in &self.blocks {
            h = block.forward(&h, cond)?;
        }
        let h = self.conv_post.forward(&leaky_relu(&h, 0.2)?)?;
        h.tanh()
    }
}
