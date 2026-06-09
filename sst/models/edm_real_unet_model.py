import torch

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_edm_real_unet_model import SREdmUNetRealModel
import torch.nn.functional as F


@MODEL_REGISTRY.register()
class EdmUNetRealModel(SREdmUNetRealModel):

    def test(self,):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        train_opt = self.opt['train']
        diffusion_opt = train_opt.get('consistency_opt')
        if diffusion_opt is None:
            raise ValueError('EdmUNetRealModel.test requires train.consistency_opt.')
        sigma_max = diffusion_opt['sigma_max']
        sf = self.opt['network_g']['scale']
        sigma = sigma_max
        sigma = torch.as_tensor(sigma).to(self.device)
        ori_h, ori_w = self.lq.shape[2:]
        # Keep validation inputs aligned to the UNet down/up hierarchy to avoid
        # one-pixel mismatches at skip concatenation for odd resolutions.
        desired_min_size = int(self.opt.get('val', {}).get('min_size', 64))
        pad_h = (desired_min_size - ori_h % desired_min_size) % desired_min_size
        pad_w = (desired_min_size - ori_w % desired_min_size) % desired_min_size
        if pad_h or pad_w:
            self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
            
        # if self.lq.shape[2] > chop_size or self.lq.shape[3] > chop_size:
            
        #     im_spliter = ImageSpliterTh(
        #                 self.lq,
        #                 chop_size,
        #                 stride=chop_stride,
        #                 sf=sf,
        #                 extra_bs=1,
        #                 )
        #     start_event.record()
        #     for im_lq_pch, index_infos in im_spliter:
        #         im_lq_up_pch = F.interpolate(im_lq_pch, scale_factor=sf, mode='bicubic')
                
        #         latent = torch.randn_like(im_lq_up_pch, device=self.device)
        #         input = im_lq_up_pch + sigma * latent.to(torch.float32)
                
        #         with torch.no_grad():
        #             im_sr_pch = self.sample_func(input, im_lq_pch, im_lq_up_pch, sigma, use_enc)     # 1 x c x h x w, [-1, 1]
        #         im_spliter.update(im_sr_pch, index_infos)
        #     im_sr_tensor = im_spliter.gather()
        #     end_event.record()
        # else:
        lq_up = F.interpolate(self.lq, scale_factor=sf, mode='bicubic')
        latent = torch.randn_like(lq_up, device=self.device)
        input = lq_up + sigma * latent
        start_event.record()
        with torch.no_grad():
            im_sr_tensor = self.sample_func(input, lq=self.lq, lq_up=lq_up, sigma=sigma)
        end_event.record()

        torch.cuda.synchronize()
        self.inference_time = start_event.elapsed_time(end_event)
        
        self.output = im_sr_tensor[:, :, :ori_h * sf, :ori_w * sf]

    def sample_func(self, input, lq, lq_up, sigma):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            
            with torch.no_grad():
                sr_tensor = self.net_g_ema(input, sigma=sigma, lq=lq, lq_up=lq_up)

        else:
            self.net_g.eval()
            with torch.no_grad():
                start_event.record()
                sr_tensor = self.net_g(input, sigma=sigma, lq=lq, lq_up=lq_up)
                end_event.record()
                torch.cuda.synchronize()
            self.net_g.train()

        return sr_tensor
