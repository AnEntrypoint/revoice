use candle_core::{bail, DType, Error, Result, Tensor};
use candle_nn::Conv1d;

use crate::gemm::gemm_dtype;

const IM2COL_TILE_ELEMENTS: usize = 1 << 25;

fn phase(j: usize, padding: usize, stride: usize) -> (isize, usize) {
    let d = j as isize - padding as isize;
    (
        d.div_euclid(stride as isize),
        d.rem_euclid(stride as isize) as usize,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Zeros,
    Reflect,
}

fn pad_reflect1d(x: &Tensor, padding: usize) -> Result<Tensor> {
    let len = x.dim(1)?;
    if padding == 0 || len < 2 {
        return x.pad_with_zeros(1, padding, padding);
    }
    let mut parts = Vec::with_capacity(2 * padding + 1);
    for i in 0..padding {
        parts.push(x.narrow(1, (padding - i).min(len - 1), 1)?);
    }
    parts.push(x.clone());
    for i in 0..padding {
        parts.push(x.narrow(1, (len - 2).saturating_sub(i), 1)?);
    }
    Tensor::cat(&parts, 1)
}

pub struct FastConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    dtype: DType,
    ksize: usize,
    c_in: usize,
    dilation: usize,
    padding: usize,
    pad_mode: PadMode,
}

impl FastConv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        ksize: usize,
        dilation: usize,
        padding: usize,
        pad_mode: PadMode,
    ) -> Result<Self> {
        let weight = if weight.is_contiguous() {
            weight
        } else {
            weight.contiguous()?
        };
        let (c_out, c_in, _) = weight.dims3()?;
        let dtype = gemm_dtype();
        let weight = weight.reshape((c_out, c_in * ksize))?;
        let weight = if dtype == DType::F32 {
            weight
        } else {
            weight.to_dtype(dtype)?
        };
        Ok(Self {
            weight,
            bias,
            dtype,
            ksize,
            c_in,
            dilation,
            padding,
            pad_mode,
        })
    }

    pub fn from_conv1d(conv: &Conv1d) -> Result<Self> {
        Self::with_pad_mode(conv, PadMode::Zeros)
    }

    pub fn with_pad_mode(conv: &Conv1d, pad_mode: PadMode) -> Result<Self> {
        let cfg = conv.config();
        if cfg.groups != 1 || cfg.stride != 1 {
            candle_core::bail!(
                "FastConv1d needs stride 1 and a single group, got stride {} groups {}",
                cfg.stride,
                cfg.groups
            );
        }
        Self::new(
            conv.weight().clone(),
            conv.bias().cloned(),
            conv.weight().dims3()?.2,
            cfg.dilation,
            cfg.padding,
            pad_mode,
        )
    }

    pub fn stack_1x1(convs: &[FastConv1d]) -> Result<FastConv1d> {
        let first = convs
            .first()
            .ok_or_else(|| Error::Msg("stack_1x1 needs at least one layer".into()))?;
        for conv in convs {
            if conv.ksize != 1 || conv.dilation != 1 || conv.padding != 0 {
                bail!("stack_1x1 needs unpadded undilated kernel-1 layers");
            }
            if conv.dtype != first.dtype {
                bail!("stack_1x1 needs layers of one dtype");
            }
        }
        let weights: Vec<Tensor> = convs.iter().map(|c| c.weight.clone()).collect();
        let weight = if weights.len() == 1 {
            weights.into_iter().next().unwrap()
        } else {
            Tensor::cat(&weights, 0)?
        };
        let biases = convs
            .iter()
            .map(|c| c.bias.as_ref())
            .collect::<Option<Vec<_>>>()
            .map(|bs| -> Result<Tensor> {
                let bs: Vec<Tensor> = bs.into_iter().cloned().collect();
                if bs.len() == 1 {
                    Ok(bs.into_iter().next().unwrap())
                } else {
                    Tensor::cat(&bs, 0)
                }
            })
            .transpose()?;
        Ok(FastConv1d {
            weight,
            bias: biases,
            dtype: first.dtype,
            ksize: 1,
            c_in: first.c_in,
            dilation: 1,
            padding: 0,
            pad_mode: PadMode::Zeros,
        })
    }

    fn im2col(&self, src: &Tensor, out_len: usize) -> Result<Tensor> {
        if self.ksize == 1 {
            return Ok(src.clone());
        }
        let c_in = src.dim(0)?;
        let taps = (0..self.ksize)
            .map(|i| src.narrow(1, i * self.dilation, out_len)?.unsqueeze(1))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&taps, 1)?.reshape((c_in * self.ksize, out_len))
    }

    fn convolve(&self, src: &Tensor, out_len: usize) -> Result<Tensor> {
        let src = if self.dtype == DType::F32 {
            src.clone()
        } else {
            src.to_dtype(self.dtype)?
        };
        let y = self.weight.matmul(&self.im2col(&src, out_len)?)?;
        let y = if self.dtype == DType::F32 {
            y
        } else {
            y.to_dtype(DType::F32)?
        };
        match &self.bias {
            Some(bias) => y.broadcast_add(&bias.reshape((bias.elem_count(), 1))?),
            None => Ok(y),
        }
    }

    fn convolve_frame_major(&self, src: &Tensor, out_len: usize) -> Result<Tensor> {
        let src = if self.dtype == DType::F32 {
            src.clone()
        } else {
            src.to_dtype(self.dtype)?
        };
        let y = self.im2col(&src, out_len)?.t()?.matmul(&self.weight.t()?)?;
        let y = if self.dtype == DType::F32 {
            y
        } else {
            y.to_dtype(DType::F32)?
        };
        match &self.bias {
            Some(bias) => y.broadcast_add(&bias.reshape((1, bias.elem_count()))?),
            None => Ok(y),
        }
    }

    fn frames_per_tile(&self) -> usize {
        (IM2COL_TILE_ELEMENTS / (self.c_in * self.ksize).max(1)).max(1)
    }

    fn tiled(&self, src: &Tensor, out_len: usize, frame_major: bool) -> Result<Vec<Tensor>> {
        let tile = self.frames_per_tile();
        let reach = self.dilation * (self.ksize - 1);
        let mut tiles = Vec::with_capacity(out_len.div_ceil(tile));
        let mut start = 0usize;
        while start < out_len {
            let n = tile.min(out_len - start);
            let window = src.narrow(1, start, n + reach)?;
            tiles.push(if frame_major {
                self.convolve_frame_major(&window, n)?
            } else {
                self.convolve(&window, n)?
            });
            start += n;
        }
        Ok(tiles)
    }

    fn padded(&self, x: &Tensor, len: usize) -> Result<(Tensor, usize)> {
        let src = x.squeeze(0)?;
        let src = if self.padding > 0 {
            match self.pad_mode {
                PadMode::Reflect => pad_reflect1d(&src, self.padding)?,
                PadMode::Zeros => src.pad_with_zeros(1, self.padding, self.padding)?,
            }
        } else {
            src
        };
        let out_len = len + 2 * self.padding - self.dilation * (self.ksize - 1);
        Ok((src, out_len))
    }

    pub fn forward_frame_major(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _c_in, len) = x.dims3()?;
        if batch != 1 {
            candle_core::bail!("FastConv1d needs a batch of one, got {batch}");
        }
        let (src, out_len) = self.padded(x, len)?;
        if out_len <= self.frames_per_tile() {
            return self.convolve_frame_major(&src, out_len)?.unsqueeze(0);
        }
        Tensor::cat(&self.tiled(&src, out_len, true)?, 0)?.unsqueeze(0)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _c_in, len) = x.dims3()?;
        if batch != 1 {
            candle_core::bail!("FastConv1d needs a batch of one, got {batch}");
        }
        let (src, out_len) = self.padded(x, len)?;
        if out_len <= self.frames_per_tile() {
            return self.convolve(&src, out_len)?.unsqueeze(0);
        }
        Tensor::cat(&self.tiled(&src, out_len, false)?, 1)?.unsqueeze(0)
    }
}

pub struct FastConvTranspose1d {
    phases: Vec<(Tensor, Vec<isize>)>,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    ksize: usize,
    c_out: usize,
}

impl FastConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Self> {
        let weight = weight.contiguous()?;
        let (_, c_out, ksize) = weight.dims3()?;
        let mut phases = Vec::with_capacity(stride);
        for r in 0..stride {
            let taps = (0..ksize)
                .filter(|&j| phase(j, padding, stride).1 == r)
                .collect::<Vec<_>>();
            if taps.is_empty() {
                bail!("transposed convolution leaves output phase {r} empty");
            }
            let mats = taps
                .iter()
                .map(|&j| weight.narrow(2, j, 1)?.squeeze(2)?.t()?.contiguous())
                .collect::<Result<Vec<_>>>()?;
            let mat = if mats.len() == 1 {
                mats.into_iter().next().unwrap()
            } else {
                Tensor::cat(&mats, 1)?
            };
            let shifts = taps
                .iter()
                .map(|&j| phase(j, padding, stride).0)
                .collect::<Vec<_>>();
            phases.push((mat, shifts));
        }
        Ok(Self {
            phases,
            bias,
            stride,
            padding,
            output_padding,
            ksize,
            c_out,
        })
    }

    pub fn out_len(&self, l_in: usize) -> usize {
        (l_in - 1) * self.stride - 2 * self.padding + self.ksize - 1 + self.output_padding + 1
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _c_in, l_in) = x.dims3()?;
        if batch != 1 {
            bail!("FastConvTranspose1d needs a batch of one, got {batch}");
        }
        let out_len = self.out_len(l_in);
        let blocks = out_len.div_ceil(self.stride);
        let lo = self
            .phases
            .iter()
            .flat_map(|(_, shifts)| shifts.iter())
            .copied()
            .min()
            .unwrap_or(0);
        let hi = self
            .phases
            .iter()
            .flat_map(|(_, shifts)| shifts.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let left = hi.max(0) as usize;
        let right = (blocks as isize - lo - l_in as isize).max(0) as usize;
        let src = x.squeeze(0)?.pad_with_zeros(1, left, right)?;

        let mut ys = Vec::with_capacity(self.stride);
        for (mat, shifts) in &self.phases {
            let vs = shifts
                .iter()
                .map(|&shift| src.narrow(1, (left as isize - shift) as usize, blocks))
                .collect::<Result<Vec<_>>>()?;
            let v = if vs.len() == 1 {
                vs.into_iter().next().unwrap()
            } else {
                Tensor::cat(&vs, 0)?
            };
            ys.push(mat.matmul(&v)?);
        }

        let out = Tensor::stack(&ys, 2)?
            .reshape((self.c_out, blocks * self.stride))?
            .narrow(1, 0, out_len)?;
        let out = match &self.bias {
            Some(bias) => out.broadcast_add(&bias.reshape((self.c_out, 1))?)?,
            None => out,
        };
        out.unsqueeze(0)
    }
}

pub fn depthwise_correlate(
    src: &Tensor,
    w: &Tensor,
    stride: usize,
    out_len: usize,
    left: usize,
) -> Result<Tensor> {
    let (c, len) = src.dims2()?;
    let (wc, ksize) = w.dims2()?;
    if wc != c {
        bail!("depthwise kernel has {wc} channels for a {c} channel input");
    }
    let tail = (stride - len % stride) % stride;
    let src = if tail == 0 {
        src.clone()
    } else {
        src.pad_with_zeros(1, 0, tail)?
    };
    let blocks = src.reshape((c, (len + tail) / stride, stride))?;
    let mut acc: Option<Tensor> = None;
    for j in 0..ksize {
        let at = left + j;
        let tap = blocks
            .narrow(2, at % stride, 1)?
            .squeeze(2)?
            .narrow(1, at / stride, out_len)?;
        let term = tap.broadcast_mul(&w.narrow(1, j, 1)?)?;
        acc = Some(match acc {
            Some(prev) => (prev + term)?,
            None => term,
        });
    }
    acc.ok_or_else(|| Error::Msg("depthwise kernel has no taps".into()))
}

pub fn depthwise_transpose(
    src: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    out_len: usize,
) -> Result<Tensor> {
    let (c, l_in) = src.dims2()?;
    let (wc, ksize) = w.dims2()?;
    if wc != c {
        bail!("depthwise kernel has {wc} channels for a {c} channel input");
    }
    let blocks = out_len.div_ceil(stride);
    let lo = (0..ksize)
        .map(|j| phase(j, padding, stride).0)
        .min()
        .unwrap_or(0);
    let hi = (0..ksize)
        .map(|j| phase(j, padding, stride).0)
        .max()
        .unwrap_or(0);
    let left = hi.max(0) as usize;
    let right = (blocks as isize - lo - l_in as isize).max(0) as usize;
    let src = src.pad_with_zeros(1, left, right)?;

    let mut ys: Vec<Option<Tensor>> = vec![None; stride];
    for j in 0..ksize {
        let (shift, r) = phase(j, padding, stride);
        let tap = src.narrow(1, (left as isize - shift) as usize, blocks)?;
        let term = tap.broadcast_mul(&w.narrow(1, j, 1)?)?;
        ys[r] = Some(match ys[r].take() {
            Some(prev) => (prev + term)?,
            None => term,
        });
    }
    let ys = ys
        .into_iter()
        .enumerate()
        .map(|(r, y)| {
            y.ok_or_else(|| Error::Msg(format!("transposed convolution leaves phase {r} empty")))
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&ys, 2)?
        .reshape((c, blocks * stride))?
        .narrow(1, 0, out_len)
}
