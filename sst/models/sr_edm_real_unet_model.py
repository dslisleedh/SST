import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import random
from contextlib import nullcontext

import pyiqa

import torch.nn.functional as F
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
# from basicsr.archs.edm_unet_arch import EDMUNet
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
# from basicsr.utils import generate_lq
# from .base_model import BaseModel
from basicsr.models.base_model import BaseModel
import torch.distributed as dist
from torchvision.transforms.functional import normalize
import lpips
import numpy as np

from sst.utils.diffusion import improved_timesteps_schedule_decrease_linear, improved_timesteps_schedule_increase_linear, karras_schedule, lognormal_timestep_distribution, q_sample


@MODEL_REGISTRY.register()
class SREdmUNetRealModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SREdmUNetRealModel, self).__init__(opt)

        self.opt = opt
        self.degradation = self.opt['degradation']
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        if self.opt.get('compile_network', False):
            if hasattr(torch, 'compile'):
                self.net_g = torch.compile(self.net_g)
            else:
                logger = get_root_logger()
                logger.warning('compile_network=True but torch.compile is unavailable in this PyTorch build. Skip compile.')

        self.lpips_dist = lpips.LPIPS(net='vgg').cuda()
        self.l1_loss = F.l1_loss
        self.l2_loss = F.mse_loss
        self._pyiqa_metric_specs = {
            'calculate_psnr': {
                'metric_name': 'psnr',
                'paired': True,
                'kwargs': {'test_y_channel': True, 'color_space': 'ycbcr'}
            },
            'calculate_ssim': {
                'metric_name': 'ssim',
                'paired': True,
                'kwargs': {'test_y_channel': True, 'color_space': 'ycbcr'}
            },
            'calculate_lpips': {'metric_name': 'lpips', 'paired': True},
            'calculate_clipiqa': {'metric_name': 'clipiqa', 'paired': False},
            'calculate_musiq': {'metric_name': 'musiq', 'paired': False},
            'calculate_niqe': {'metric_name': 'niqe', 'paired': False},
            'calculate_maniqa': {'metric_name': 'maniqa', 'paired': False},
            'calculate_liqe': {'metric_name': 'liqe', 'paired': False},
            'calculate_liqe_mix': {'metric_name': 'liqe_mix', 'paired': False},
            'calculate_topiq_nr': {'metric_name': 'topiq_nr', 'paired': False},
            'calculate_topiq_nr_flive': {'metric_name': 'topiq_nr-flive', 'paired': False},
            'calculate_topiq_nr_spaq': {'metric_name': 'topiq_nr-spaq', 'paired': False},
            'calculate_topiq_fr': {'metric_name': 'topiq_fr', 'paired': True},
            'calculate_dists': {'metric_name': 'dists', 'paired': True},
            'calculate_fid': {'metric_name': 'fid', 'paired': False},
            'calculate_ahiq': {'metric_name': 'ahiq', 'paired': True},
        }
        self._pyiqa_metrics = {}
        self._build_configured_val_metrics()
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        # self.net_g = torch.compile(self.net_g) # torch2.0
        if opt['train'].get('align_opt') is not None:
            ckpt_path = opt.get('dinotxt_backbone_weight_path', None)
            model_name = 'dinov3_vitl16'
            self.backbone = torch.hub.load(
                REPO_DIR,
                model_name,
                source='local',
                weights=ckpt_path
            )
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone = self.backbone.to(self.device)
            self.dinotxt_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.dinotxt_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

            align_layer = -1
            self.hook = []
            h = self.backbone.blocks[align_layer].register_forward_hook(self.hook_target_layer)
            self.hook.append(h)

    def hook_target_layer(self, module, input, output):
        output = output if isinstance(output, torch.Tensor) else output[0]
        self.align_feat = output[:, 5:]  # remove non-patch tokens

    def _build_configured_val_metrics(self):
        val_metrics = self.opt.get('val', {}).get('metrics') or {}
        for metric_opt in val_metrics.values():
            metric_type = metric_opt.get('type')
            if metric_type in self._pyiqa_metric_specs:
                self._get_pyiqa_metric(metric_type)

    def _get_pyiqa_metric(self, metric_type):
        metric = self._pyiqa_metrics.get(metric_type)
        if metric is not None:
            return metric

        spec = self._pyiqa_metric_specs.get(metric_type)
        if spec is None:
            raise NotImplementedError(
                f'Validation metric {metric_type} is not supported in {self.__class__.__name__}.'
            )

        metric = pyiqa.create_metric(spec['metric_name'], device=self.device, **spec.get('kwargs', {}))
        self._pyiqa_metrics[metric_type] = metric
        return metric

    def _calculate_validation_metric(self, metric_opt, sr_tensor, gt_tensor=None):
        metric_type = metric_opt.get('type')
        metric = self._get_pyiqa_metric(metric_type)
        spec = self._pyiqa_metric_specs[metric_type]

        if spec['paired']:
            if gt_tensor is None:
                raise ValueError(f'Validation metric {metric_type} requires gt images.')
            score = metric(sr_tensor, gt_tensor)
        else:
            score = metric(sr_tensor)

        return score.mean().item()

    def prepare_data(self, data, dtype=torch.float32, realesrgan=True, phase='train'):
        # if realesrgan is None:
        #     realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.degradation['sf'], int):
                sf = self.degradation['sf']
            else:
                assert len(self.degradation['sf']) == 2
                sf = random.uniform(*self.degradation['sf'])

            if self.degradation['use_sharp']:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.degradation['gray_noise_prob']
            if random.random() < self.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.degradation['second_order_prob']:
                # blur
                if random.random() < self.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.degradation['gray_noise_prob2']
                if random.random() < self.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
            
            # im_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            
            # resize back
            if self.degradation['resize_back']:
                img_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                img_lq_up = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                # temp_sf = self.degradation['sf']
            else:
                img_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                # temp_sf = self.degradation['sf']
                img_lq_up = None


            # random crop
            # gt_size = self.configs.degradation['gt_size']
            # im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            mean = data['mean'][0]
            std = data['std'][0]
            img_lq_ori = normalize(img_lq_ori, mean, std, inplace=True)
            img_lq_up = normalize(img_lq_up, mean, std, inplace=True)  # [0, 1] to [-1, 1]
            img_gt_non_normalized = im_gt.clone()
            im_gt = normalize(im_gt, mean, std, inplace=True)  # [0, 1] to [-1, 1]

            return {'lq':img_lq_ori, 'gt':im_gt, 'gt_non_normalized': img_gt_non_normalized, 'lq_up': img_lq_up if img_lq_up is not None else None}
        else:
            lq = data['lq'].cuda()
            if 'gt' in data:
                return {'lq': lq, 'gt': data['gt']}
            else:
                return {'lq': lq}
            # if self.degradation['resize_back']:
            #     lq_up = F.interpolate(lq, scale_factor=self.degradation['sf'], mode='bicubic')
            #     return {'lq':lq, 'gt':data['gt'], 'lq_up':lq_up}
            # return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def init_training_settings(self):
        # self.net_g.train()
        self.net_g.train()
        train_opt = self.opt['train']
        self.use_amp = self.opt.get('use_amp', train_opt.get('use_amp', True))
        self.amp_dtype = torch.bfloat16
        self.amp_enabled = False
        if self.use_amp:
            logger = get_root_logger()
            if self.device.type != 'cuda':
                logger.warning('AMP bf16 was requested but device is not CUDA. Disable AMP.')
            elif not torch.cuda.is_bf16_supported():
                logger.warning('AMP bf16 was requested but current CUDA device does not support bf16. Disable AMP.')
            else:
                self.amp_enabled = True
                logger.info('Use AMP training with bfloat16 autocast (without GradScaler).')

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('consistency_opt'):
            self.consistency_opt = train_opt['consistency_opt']
            # loss_opt = {}
            # loss_opt['loss_weight'] = self.consistency_opt['loss_weight']

            self.net_target = build_network(self.opt['network_g']).to(self.device)
            self.net_target.eval()
            for param in self.net_target.parameters():
                param.requires_grad = False
            for p_target, p_net in zip(self.net_target.parameters(), self.net_g.parameters()):
                p_target.copy_(p_net.detach())        
        else:
            self.consistency_opt = None

        if train_opt.get('align_opt'):
            self.cri_align = build_loss(train_opt['align_opt']).to(self.device)
        else:
            self.cri_align = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        weight_decay = float(train_opt['optim_g'].get('weight_decay', 0) or 0)
        if 'weight_decay' in train_opt['optim_g']:
            train_opt['optim_g']['weight_decay'] = weight_decay
        optim_params = self._get_optim_params(weight_decay)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def _get_optim_params(self, weight_decay):
        net_g = self.get_bare_model(self.net_g)
        logger = get_root_logger()

        if weight_decay <= 0:
            optim_params = []
            for k, v in net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning(f'Params {k} will not be optimized.')
            return optim_params

        no_decay_names, no_decay_keywords = self._collect_no_weight_decay_rules(net_g)

        decay_params = []
        no_decay_params = []
        no_decay_param_names = []
        for k, v in net_g.named_parameters():
            if not v.requires_grad:
                logger.warning(f'Params {k} will not be optimized.')
                continue

            if self._use_no_weight_decay(k, v, no_decay_names, no_decay_keywords):
                no_decay_params.append(v)
                no_decay_param_names.append(k)
            else:
                decay_params.append(v)

        optim_params = []
        if decay_params:
            optim_params.append({'params': decay_params})
        if no_decay_params:
            optim_params.append({'params': no_decay_params, 'weight_decay': 0.0})

        logger.info(
            f'Use weight decay filtering: '
            f'{len(decay_params)} decay params, {len(no_decay_params)} no_decay params.'
        )
        if no_decay_param_names:
            logger.info('No weight decay params:\n  ' + '\n  '.join(no_decay_param_names))
        return optim_params

    @staticmethod
    def _collect_no_weight_decay_rules(net):
        no_decay_names = set()
        no_decay_keywords = set()
        for module_prefix, module in net.named_modules():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay() or ():
                    full_name = f'{module_prefix}.{name}' if module_prefix else name
                    no_decay_names.add(full_name)
            if hasattr(module, 'no_weight_decay_keywords'):
                no_decay_keywords.update(module.no_weight_decay_keywords() or ())
        return no_decay_names, no_decay_keywords

    @staticmethod
    def _use_no_weight_decay(param_name, param, no_decay_names, no_decay_keywords):
        return (
            param_name in no_decay_names
            or any(keyword in param_name for keyword in no_decay_keywords)
        )

    def feed_data(self, data):
        if 'phase' in data:
            phase = data['phase']
            data = self.prepare_data(data, phase=phase)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None
        # Keep the input tensor in a standard contiguous layout to avoid DDP grad layout warnings.
        self.lq = data['lq'].to(self.device).contiguous()
        if 'gt_non_normalized' in data:
            self.gt_non_normalized = data['gt_non_normalized'].to(self.device)
        if 'lq_up' in data:
            self.lq_up = data['lq_up'].to(self.device)

    def optimize_parameters(self, current_iter,):        
        self.optimizer_g.zero_grad()
        amp_context = torch.autocast(device_type='cuda', dtype=self.amp_dtype) if self.amp_enabled else nullcontext()
        with amp_context:
            l_total = 0
            loss_dict = OrderedDict()
        
            # consistency loss
            if self.consistency_opt:
                for p_target, p_net in zip(self.net_target.parameters(), self.net_g.parameters()):
                    p_target.copy_(p_net.detach())

                sigma_min = self.consistency_opt.get('sigma_min', 0)
                sigma_max = self.consistency_opt.get('sigma_max', 1)
                s0 = self.consistency_opt.get('s0')
                s1 = self.consistency_opt.get('s1')
                noise_rho = self.consistency_opt.get('noise_rho', 1)
                res_rho = self.consistency_opt.get('res_rho', 1)
                loss_opt = self.consistency_opt.get('loss_opt', None)
                res_min = 0
                res_max = 1
                total_iter = self.opt['train']['total_iter']
                if s0 > s1:
                    num_steps = improved_timesteps_schedule_decrease_linear(current_iter, total_iter, s0, s1,)
                elif s0 < s1:
                    num_steps = improved_timesteps_schedule_increase_linear(current_iter, total_iter, s0, s1,)
                else:
                    num_steps = s0 + 1
                num_steps = int(num_steps)
                sigmas = karras_schedule(num_steps, sigma_min, sigma_max, noise_rho, device=self.device)
                alphas = karras_schedule(num_steps, res_min, res_max, res_rho, device=self.device)
                
                timestep = torch.randint(0, num_steps - 1, size=(1,), device=self.device)
                index = timestep.repeat(self.gt.shape[0])
                next_index = index + 1

                noise = torch.randn_like(self.gt)

                x_cur = q_sample(self.lq_up, self.gt, sigmas, alphas, index, noise)
                x_next = q_sample(self.lq_up, self.gt, sigmas, alphas, next_index, noise)
                
                lq = self.lq 
                lq_up = self.lq_up
                
                with torch.no_grad():
                    _, d_f_t = self.net_target(x_cur, lq=lq, lq_up=lq_up, sigma=sigmas[index], return_pair=True)
                    d_f_t = d_f_t.detach()
                d_i, d_f = self.net_g(x_next, lq=lq, lq_up=lq_up, sigma=sigmas[next_index],)

                self.output = d_f

                l_consistency_i = 0.0
                for name, l_opt in loss_opt.items():
                    loss = self.get_loss_func(l_opt, d_i, d_f_t).mean()
                    l_consistency_i += loss
                l_consistency_i = l_consistency_i * 0.25  # following original paper.

                loss_dict['l_consistency_i'] = l_consistency_i
                l_total += l_consistency_i

                l_consistency_f = 0.0
                for name, l_opt in loss_opt.items():
                    loss = self.get_loss_func(l_opt, d_f, d_f_t).mean()
                    l_consistency_f += loss

                loss_dict['l_consistency_f'] = l_consistency_f
                l_total += l_consistency_f

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # clear cache
        torch.cuda.empty_cache()

    def sum_gradients(self, model):
        total_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad += torch.sum(torch.abs(param.grad)).item()
            else:
                print('none')
        return total_grad
    
    def sum_params(self, model):
        total_params = 0.0
        for param in model.parameters():
            total_params += param.sum().item()
        return total_params


    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                netg = self.get_bare_model(self.net_g_ema)
                self.output = netg(self.lq)
                # self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                netg = self.get_bare_model(self.net_g)
                self.output = netg(self.lq)
                # self.output = self.net_g(self.lq)
        
        self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        total_inference_time = 0.0
        num_imgs = 0
        for idx, val_data in enumerate(dataloader):
            num_imgs += 1
            if 'gt_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            else: 
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            val_data = self.prepare_data(val_data, phase='val')
            self.feed_data(val_data)
            self.test()
            total_inference_time += self.inference_time


            visuals = self.get_current_visuals()
            # print(torch.max(visuals['result']).item())
            sr_tensor = visuals['result']
            sr_img = tensor2img([visuals['result']])
            # print(torch.max(visuals['result']).item())

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_tensor = visuals['gt']
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += self._calculate_validation_metric(
                        opt_, sr_tensor, gt_tensor if 'gt' in visuals else None)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        # print(num_imgs)
        avg_inference_time = total_inference_time / num_imgs
        log_str = f'inference time:{avg_inference_time:.6f}s\n'
        logger = get_root_logger()
        logger.info(log_str)
        # print(f'推理时间: {avg_inference_time:.6f} 秒')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        torch.cuda.empty_cache()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        self.lq = self.totensor(self.lq)
        out_dict['lq'] = self.lq.detach().cpu()
        self.output = self.totensor(self.output)
        out_dict['result'] = self.output.detach().cpu()
        if self.gt is not None:
            self.gt = self.totensor(self.gt)
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def totensor(self, input):
        # input = input.clamp(min=-1, max=1)
        input = (input + 1) / 2
        input = input.clamp(0, 1)
        return input
    
    def replace_nan_in_batch(im_lq, im_gt):
        '''
        Input:
            im_lq, im_gt: b x c x h x w
        '''
        if torch.isnan(im_lq).sum() > 0:
            valid_index = []
            im_lq = im_lq.contiguous()
            for ii in range(im_lq.shape[0]):
                if torch.isnan(im_lq[ii,]).sum() == 0:
                    valid_index.append(ii)
            assert len(valid_index) > 0
            im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
            flag = True
        else:
            flag = False
        return im_lq, im_gt, flag
    
    def charbonnier_loss(self, pred, target, eps=1e-12):
        return torch.sqrt((pred - target)**2 + eps)

    def unwrap_network_output(self, output):
        return output if isinstance(output, torch.Tensor) else output[0]

    def sobel_map(self, tensor, use_gray=True, normalize_range=True, eps=1e-6):
        tensor = tensor.float()
        if normalize_range:
            tensor = ((tensor + 1) / 2).clamp(0, 1)

        if use_gray and tensor.shape[1] >= 3:
            gray_weight = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            tensor = (tensor[:, :3] * gray_weight).sum(dim=1, keepdim=True)
        elif use_gray:
            tensor = tensor.mean(dim=1, keepdim=True)

        channels = tensor.shape[1]
        kernel_x = tensor.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
        kernel_y = tensor.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(tensor, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        grad_y = F.conv2d(tensor, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)

    def get_loss_func(self, loss_opt, input1, input2):
        loss_type = loss_opt['type']
        weight = loss_opt['weight']
        if loss_type == 'l2':
            loss = self.l2_loss(input1, input2)
        elif loss_type == 'l1':
            loss = self.l1_loss(input1, input2)
        elif loss_type == 'lpips':
            if input1.shape[1] == 1:
                input1 = input1.repeat(1, 3, 1, 1)
                input2 = input2.repeat(1, 3, 1, 1)
            loss = self.lpips_dist(input1, input2)
        elif loss_type == 'charbonnier':
            loss = self.charbonnier_loss(input1, input2)
        else:
            raise ValueError(f'Unsupported loss function: {loss_type}.')

        return weight * loss
    
    def get_diff_loss(self, diff_loss_opt, ):
        loss_type = diff_loss_opt['loss_type']
        return
