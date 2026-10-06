use candle_core::{Device, Result, Tensor};
use candle_nn::{conv1d, Conv1dConfig, VarBuilder};

use crate::fastconv::FastConv1d;

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
    dconv: FastConv1d,
    out_conv: FastConv1d,
    hidden_dim: usize,
}

impl WnLayer {
    fn new(hidden_dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: dilation,
            dilation,
            ..Default::default()
        };
        let dconv = conv1d(hidden_dim, hidden_dim * 2, 3, cfg, vb.pp("dconv"))?;
        let cfg1 = Conv1dConfig::default();
        let out_conv = conv1d(hidden_dim, hidden_dim * 2, 1, cfg1, vb.pp("out"))?;
        Ok(Self {
            dconv: FastConv1d::from_conv1d(&dconv)?,
            out_conv: FastConv1d::from_conv1d(&out_conv)?,
            hidden_dim,
        })
    }

    fn forward(&self, z: &Tensor, local: &Tensor, gate: &Tensor) -> Result<(Tensor, Tensor)> {
        let identity = z.clone();
        let z = z.broadcast_add(gate)?;
        let z = self.dconv.forward(&z)?;
        crate::profile::tick_at(3, z.device(), "    cfm.layer.dconv");
        let z = (z + local)?;

        let a = z.narrow(1, 0, self.hidden_dim)?.tanh()?;
        let b = candle_nn::ops::sigmoid(&z.narrow(1, self.hidden_dim, self.hidden_dim)?)?;
        let z = (a * b)?;

        let h = self.out_conv.forward(&z)?;
        crate::profile::tick_at(3, z.device(), "    cfm.layer.outconv");
        let z_out = h.narrow(1, 0, self.hidden_dim)?;
        let s = h.narrow(1, self.hidden_dim, self.hidden_dim)?;
        let o = ((z_out + identity)? / (2f64.sqrt()))?;
        Ok((o, s))
    }
}

pub struct WaveNet {
    start: FastConv1d,
    gate_conv: FastConv1d,
    local_conv: FastConv1d,
    layers: Vec<WnLayer>,
    end: FastConv1d,
    hidden_dim: usize,
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
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_layers);
        let mut gate_convs = Vec::with_capacity(n_layers);
        let mut local_convs = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dilation = 1usize << (i % dilation_cycle);
            let layer_vb = layers_vb.pp(i);
            let gconv = conv1d(global_dim, hidden_dim, 1, cfg1, layer_vb.pp("gconv"))?;
            let lconv = conv1d(local_dim, hidden_dim * 2, 1, cfg1, layer_vb.pp("lconv"))?;
            gate_convs.push(FastConv1d::from_conv1d(&gconv)?);
            local_convs.push(FastConv1d::from_conv1d(&lconv)?);
            layers.push(WnLayer::new(hidden_dim, dilation, layer_vb)?);
        }
        let end = conv1d(hidden_dim, output_dim, 1, cfg1, vb.pp("end"))?;
        Ok(Self {
            start: FastConv1d::from_conv1d(&start)?,
            gate_conv: FastConv1d::stack_1x1(&gate_convs)?,
            local_conv: FastConv1d::stack_1x1(&local_convs)?,
            layers,
            end: FastConv1d::from_conv1d(&end)?,
            hidden_dim,
            time_dim: global_dim,
            device: device.clone(),
        })
    }

    fn embed_time(&self, t: &Tensor) -> Result<Tensor> {
        let emb = sinusoidal_time_embedding(t, self.time_dim, &self.device)?;
        emb.reshape((emb.dim(0)?, emb.dim(1)?, 1))
    }

    fn stacked_local_conditioning(&self, cond: &Tensor) -> Result<Tensor> {
        self.local_conv.forward(cond)
    }

    fn forward_prepared(&self, x: &Tensor, cond: &Tensor, t: &Tensor) -> Result<Tensor> {
        let cond = instance_norm1d(cond)?;
        let local = self.stacked_local_conditioning(&cond)?;
        self.forward_normed(x, &local, t)
    }

    fn forward_normed(&self, x: &Tensor, local: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let mut z = self.start.forward(x)?;
        let gate = self.gate_conv.forward(time_emb)?;
        let mut skip_sum: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let gate_i = gate.narrow(1, i * self.hidden_dim, self.hidden_dim)?;
            let local_i = local.narrow(1, i * self.hidden_dim * 2, self.hidden_dim * 2)?;
            let (new_z, skip) = layer.forward(&z, &local_i, &gate_i)?;
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

    pub fn forward(&self, x: &Tensor, cond: &Tensor, t: &Tensor) -> Result<Tensor> {
        self.forward_prepared(x, cond, t)
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

        let cond = instance_norm1d(cond)?;
        let local = wn.stacked_local_conditioning(&cond)?;
        crate::profile::tick(device, "cfm.local_cond");

        for i in 0..self.n_steps {
            let t0 = mapped_times[i];
            let t1 = mapped_times[i + 1];
            let dt = t1 - t0;
            let t_mid = t0 + dt / 2.0;

            let t0_tensor = Tensor::from_vec(vec![t0 as f32], 1, device)?;
            let emb0 = wn.embed_time(&t0_tensor)?;
            let v0 = wn.forward_normed(&x, &local, &emb0)?;
            crate::profile::tick(device, &format!("cfm.step{i}.v0"));

            let x_mid = (&x + (v0 * (dt / 2.0))?)?;
            let t_mid_tensor = Tensor::from_vec(vec![t_mid as f32], 1, device)?;
            let emb_mid = wn.embed_time(&t_mid_tensor)?;
            let v_mid = wn.forward_normed(&x_mid, &local, &emb_mid)?;
            crate::profile::tick(device, &format!("cfm.step{i}.vmid"));

            x = (&x + (v_mid * dt)?)?;
        }
        crate::profile::tick(device, "cfm.done");
        Ok(x)
    }
}
