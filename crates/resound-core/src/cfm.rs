use candle_core::{Device, Result, Tensor};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, Module, VarBuilder};

fn sinusoidal_time_embedding(t: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let powers: Vec<f32> = (0..half)
        .map(|i| 10f32.powf(4.0 * i as f32 / half as f32))
        .collect();
    let powers = Tensor::from_vec(powers, half, device)?;
    let t = t.reshape((t.elem_count(), 1))?;
    let args = t.broadcast_mul(&powers.reshape((1, half))?)?;
    let sin = args.sin()?;
    let cos = args.cos()?;
    Tensor::cat(&[sin, cos], 1)
}

fn instance_norm1d(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    centered.broadcast_div(&(var + 1e-5)?.sqrt()?)
}

struct WnLayer {
    gconv: Conv1d,
    lconv: Conv1d,
    dconv: Conv1d,
    out_conv: Conv1d,
    hidden_dim: usize,
}

impl WnLayer {
    fn new(hidden_dim: usize, local_dim: usize, global_dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let cfg1 = Conv1dConfig::default();
        let gconv = conv1d(global_dim, hidden_dim, 1, cfg1, vb.pp("gconv"))?;
        let lconv = conv1d(local_dim, hidden_dim * 2, 1, cfg1, vb.pp("lconv"))?;
        let cfg = Conv1dConfig {
            padding: dilation,
            dilation,
            ..Default::default()
        };
        let dconv = conv1d(hidden_dim, hidden_dim * 2, 3, cfg, vb.pp("dconv"))?;
        let out_conv = conv1d(hidden_dim, hidden_dim * 2, 1, cfg1, vb.pp("out"))?;
        Ok(Self {
            gconv,
            lconv,
            dconv,
            out_conv,
            hidden_dim,
        })
    }

    fn forward(&self, z: &Tensor, l: &Tensor, g: &Tensor) -> Result<(Tensor, Tensor)> {
        let identity = z.clone();
        let z = z.broadcast_add(&self.gconv.forward(g)?)?;
        let z = self.dconv.forward(&z)?;
        let z = (z + self.lconv.forward(l)?)?;

        let a = z.narrow(1, 0, self.hidden_dim)?.tanh()?;
        let b = candle_nn::ops::sigmoid(&z.narrow(1, self.hidden_dim, self.hidden_dim)?)?;
        let z = (a * b)?;

        let h = self.out_conv.forward(&z)?;
        let z_out = h.narrow(1, 0, self.hidden_dim)?;
        let s = h.narrow(1, self.hidden_dim, self.hidden_dim)?;
        let o = ((z_out + identity)? / (2f64.sqrt()))?;
        Ok((o, s))
    }
}

pub struct WaveNet {
    start: Conv1d,
    layers: Vec<WnLayer>,
    end: Conv1d,
    time_dim: usize,
    device: Device,
}

impl WaveNet {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        local_dim: usize,
        global_dim: usize,
        n_layers: usize,
        dilation_cycle: usize,
        hidden_dim: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let cfg1 = Conv1dConfig::default();
        let start = conv1d(input_dim, hidden_dim, 1, cfg1, vb.pp("start"))?;
        let mut layers = Vec::with_capacity(n_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..n_layers {
            let dilation = 1usize << (i % dilation_cycle);
            layers.push(WnLayer::new(
                hidden_dim,
                local_dim,
                global_dim,
                dilation,
                layers_vb.pp(i),
            )?);
        }
        let end = conv1d(hidden_dim, output_dim, 1, cfg1, vb.pp("end"))?;
        Ok(Self {
            start,
            layers,
            end,
            time_dim: global_dim,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor, t: &Tensor) -> Result<Tensor> {
        let time_emb = sinusoidal_time_embedding(t, self.time_dim, &self.device)?;
        let time_emb = time_emb.reshape((time_emb.dim(0)?, time_emb.dim(1)?, 1))?;
        let cond = instance_norm1d(cond)?;

        let mut z = self.start.forward(x)?;
        let mut skip_sum: Option<Tensor> = None;
        for layer in &self.layers {
            let (new_z, skip) = layer.forward(&z, &cond, &time_emb)?;
            z = new_z;
            skip_sum = Some(match skip_sum {
                Some(acc) => (acc + skip)?,
                None => skip,
            });
        }
        let n = self.layers.len() as f64;
        let skip_sum = skip_sum.ok_or_else(|| candle_core::Error::Msg("WaveNet has zero layers".into()))?;
        let skip_sum = (skip_sum / n.sqrt())?;
        self.end.forward(&skip_sum)
    }
}

fn time_mapping(t: f64) -> f64 {
    let a = 0.08737802538415268f64;
    (a.powf(t) - 1.0) / (a - 1.0)
}

pub struct CfmSolver {
    n_steps: usize,
}

impl CfmSolver {
    pub fn new(nfe: usize) -> Self {
        CfmSolver { n_steps: nfe / 2 }
    }

    pub fn sample(&self, wn: &WaveNet, cond: &Tensor, psi0: &Tensor, device: &Device) -> Result<Tensor> {
        let mut x = psi0.clone();
        let raw_times: Vec<f64> = (0..=self.n_steps)
            .map(|i| i as f64 / self.n_steps as f64)
            .collect();
        let mapped_times: Vec<f64> = raw_times.iter().map(|&t| time_mapping(t)).collect();

        for i in 0..self.n_steps {
            let t0 = mapped_times[i];
            let t1 = mapped_times[i + 1];
            let dt = t1 - t0;
            let t_mid = t0 + dt / 2.0;

            let t0_tensor = Tensor::from_vec(vec![t0 as f32], 1, device)?;
            let v0 = wn.forward(&x, cond, &t0_tensor)?;

            let x_mid = (&x + (v0.clone() * (dt / 2.0))?)?;
            let t_mid_tensor = Tensor::from_vec(vec![t_mid as f32], 1, device)?;
            let v_mid = wn.forward(&x_mid, cond, &t_mid_tensor)?;

            x = (&x + (v_mid * dt)?)?;
        }
        Ok(x)
    }
}
