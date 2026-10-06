use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv1d, conv_transpose1d, Conv1dConfig, ConvTranspose1dConfig, VarBuilder};

use crate::aliasfree::AmpBlock;
use crate::fastconv::{FastConv1d, FastConvTranspose1d};

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    x.maximum(&(x * negative_slope)?)
}

fn as_bias(bias: Option<&Tensor>, offset: usize, len: usize) -> Option<Tensor> {
    bias.and_then(|b| b.narrow(0, offset, len).ok())
}

struct KernelPredictor {
    input_conv: FastConv1d,
    res_convs: Vec<(FastConv1d, FastConv1d)>,
    kernel_convs: Vec<FastConv1d>,
    bias_convs: Vec<FastConv1d>,
    conv_in: usize,
    conv_out: usize,
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
            res_convs.push((
                FastConv1d::from_conv1d(&c1)?,
                FastConv1d::from_conv1d(&c2)?,
            ));
        }

        let kernel_out_dim = conv_layers * conv_in * conv_out * kpnet_conv_size;
        let bias_out_dim = conv_layers * conv_out;
        let cfg3 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let kernel_conv = conv1d(kpnet_hidden, kernel_out_dim, 3, cfg3, vb.pp("kernel_conv"))?;
        let bias_conv = conv1d(kpnet_hidden, bias_out_dim, 3, cfg3, vb.pp("bias_conv"))?;

        let per_layer_kernel = conv_in * conv_out * kpnet_conv_size;
        let mut kernel_convs = Vec::with_capacity(conv_layers);
        let mut bias_convs = Vec::with_capacity(conv_layers);
        for l in 0..conv_layers {
            kernel_convs.push(FastConv1d::new(
                kernel_conv.weight().narrow(0, l * per_layer_kernel, per_layer_kernel)?,
                as_bias(kernel_conv.bias(), l * per_layer_kernel, per_layer_kernel),
                3,
                1,
                1,
            )?);
            bias_convs.push(FastConv1d::new(
                bias_conv.weight().narrow(0, l * conv_out, conv_out)?,
                as_bias(bias_conv.bias(), l * conv_out, conv_out),
                3,
                1,
                1,
            )?);
        }

        Ok(Self {
            input_conv: FastConv1d::from_conv1d(&input_conv)?,
            res_convs,
            kernel_convs,
            bias_convs,
            conv_in,
            conv_out,
            kpnet_conv_size,
        })
    }

    fn trunk(&self, cond: &Tensor) -> Result<Tensor> {
        let mut h = leaky_relu(&self.input_conv.forward(cond)?, 0.1)?;
        for (c1, c2) in &self.res_convs {
            let y = leaky_relu(&c1.forward(&h)?, 0.1)?;
            let y = leaky_relu(&c2.forward(&y)?, 0.1)?;
            h = (&h + y)?;
        }
        Ok(h)
    }

    fn kernels(&self, h: &Tensor, layer: usize) -> Result<(Tensor, Tensor)> {
        let flat = self.kernel_convs[layer].forward(h)?;
        let kernel_dtype = crate::gemm::gemm_dtype();
        let flat = if kernel_dtype == DType::F32 {
            flat
        } else {
            flat.to_dtype(kernel_dtype)?
        };
        let (b, _, t) = flat.dims3()?;
        let kernel = flat.reshape((b, self.conv_in, self.conv_out, self.kpnet_conv_size, t))?;
        let bias = self.bias_convs[layer]
            .forward(h)?
            .reshape((b, self.conv_out, t))?;
        Ok((kernel, bias))
    }
}

const LVC_TILE_SAMPLES: usize = 1 << 12;

fn location_variable_convolution(
    x: &Tensor,
    kernel: &Tensor,
    bias: &Tensor,
    dilation: usize,
    hop_size: usize,
) -> Result<Tensor> {
    let (b, x_len, c_in) = x.dims3()?;
    let (_, _, c_out, ksize, n_frames) = kernel.dims5()?;
    let half = (ksize - 1) / 2;
    let span = n_frames * hop_size;
    let tail = (ksize - 1) * dilation;

    let dev = x.device().clone();
    let src = x.pad_with_zeros(1, half * dilation, (span + tail).saturating_sub(x_len))?;
    crate::profile::tick(&dev, "    lvc.pad");

    let mut taps = Vec::with_capacity(ksize);
    let mut weights = Vec::with_capacity(ksize);
    for k in 0..ksize {
        taps.push(src.narrow(1, k * dilation, span)?.reshape((b, n_frames, hop_size, c_in))?);
        weights.push(kernel.narrow(3, k, 1)?.squeeze(3)?.permute((0, 3, 1, 2))?);
    }
    let bias_by_frame = bias.permute((0, 2, 1))?.unsqueeze(2)?;
    crate::profile::tick(&dev, "    lvc.taps");

    let tile_frames = (LVC_TILE_SAMPLES / hop_size).max(1);
    let mut tiles = Vec::with_capacity(n_frames.div_ceil(tile_frames));
    let mut fs = 0usize;
    while fs < n_frames {
        let nf = tile_frames.min(n_frames - fs);
        let mut acc: Option<Tensor> = None;
        for k in 0..ksize {
            let a = taps[k].narrow(1, fs, nf)?;
            let w = weights[k].narrow(1, fs, nf)?.contiguous()?;
            let y = if a.dtype() == w.dtype() {
                a.matmul(&w)?
            } else {
                a.to_dtype(w.dtype())?.matmul(&w)?.to_dtype(DType::F32)?
            };
            acc = Some(match acc {
                Some(prev) => (prev + y)?,
                None => y,
            });
        }
        let y = acc.ok_or_else(|| candle_core::Error::Msg("lvc kernel has no taps".into()))?;
        let y = y.broadcast_add(&bias_by_frame.narrow(1, fs, nf)?)?;
        tiles.push(y.permute((0, 3, 1, 2))?.reshape((b, c_out, nf * hop_size))?);
        fs += nf;
    }
    crate::profile::tick(&dev, "    lvc.gemm");

    let out = if tiles.len() == 1 {
        tiles.swap_remove(0)
    } else {
        Tensor::cat(&tiles, 2)?
    };
    let out_len = out.dim(2)?.min(x_len);
    out.narrow(2, 0, out_len)
}

struct LvcBlock {
    convt_pre: FastConvTranspose1d,
    amp: AmpBlock,
    conv_blocks: Vec<FastConv1d>,
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
        let padding = stride / 2 + stride % 2;
        let output_padding = stride % 2;
        let cfg = ConvTranspose1dConfig {
            padding,
            output_padding,
            stride,
            ..Default::default()
        };
        let convt_pre = conv_transpose1d(channels, channels, 2 * stride, cfg, vb.pp("convt_pre").pp("1"))?;
        let convt_pre = FastConvTranspose1d::new(
            convt_pre.weight().clone(),
            convt_pre.bias().cloned(),
            stride,
            padding,
            output_padding,
        )?;
        let amp = AmpBlock::new(channels, &[1, 3, 5], vb.pp("amp_block"), device)?;

        let mut conv_blocks = Vec::with_capacity(dilations.len());
        let cb_vb = vb.pp("conv_blocks");
        for (i, &d) in dilations.iter().enumerate() {
            let cfg = Conv1dConfig {
                padding: d,
                dilation: d,
                ..Default::default()
            };
            conv_blocks.push(FastConv1d::from_conv1d(&conv1d(
                channels, channels, 3, cfg, cb_vb.pp(i).pp("1"),
            )?)?);
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
        let dev = x.device().clone();
        let x = self.convt_pre.forward(&leaky_relu(x, 0.2)?)?;
        crate::profile::tick(&dev, "  lvc.convt_pre");
        let mut h = self.amp.forward(&x)?;
        crate::profile::tick(&dev, "  lvc.amp");

        let kp_h = self.kernel_predictor.trunk(cond)?;
        crate::profile::tick(&dev, "  lvc.kp_trunk");
        for i in 0..self.dilations.len() {
            let y = leaky_relu(&h, 0.2)?;
            let y = leaky_relu(&self.conv_blocks[i].forward_frame_major(&y)?, 0.2)?;
            crate::profile::tick(&dev, &format!("  lvc.dil{i}.conv"));
            let (kernel, bias) = self.kernel_predictor.kernels(&kp_h, i)?;
            crate::profile::tick(&dev, &format!("  lvc.dil{i}.kernels"));
            let lvc_out = location_variable_convolution(&y, &kernel, &bias, 1, self.cond_hop_length)?;
            crate::profile::tick(&dev, &format!("  lvc.dil{i}.lvc"));
            let lvc_out = lvc_out.narrow(1, 0, self.channels * 2)?;
            let gate = lvc_out.narrow(1, 0, self.channels)?;
            let filt = lvc_out.narrow(1, self.channels, self.channels)?;
            let gated = (candle_nn::ops::sigmoid(&gate)? * filt.tanh()?)?;
            let min_len = h.dim(2)?.min(gated.dim(2)?);
            h = (h.narrow(2, 0, min_len)? + gated.narrow(2, 0, min_len)?)?;
            crate::profile::tick(&dev, &format!("  lvc.dil{i}.gate"));
        }
        Ok(h)
    }
}

pub struct UnivNet {
    conv_pre: FastConv1d,
    blocks: Vec<LvcBlock>,
    conv_post: FastConv1d,
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
            conv_pre: FastConv1d::from_conv1d(&conv_pre)?,
            blocks,
            conv_post: FastConv1d::from_conv1d(&conv_post)?,
        })
    }

    pub fn forward(&self, noise: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_pre.forward(noise)?;
        let dev = h.device().clone();
        crate::profile::tick(&dev, "univnet.conv_pre");
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, cond)?;
            crate::profile::tick(&dev, &format!("univnet.block{i}"));
        }
        let h = self.conv_post.forward(&leaky_relu(&h, 0.2)?)?;
        crate::profile::tick(&dev, "univnet.conv_post");
        h.tanh()
    }
}
