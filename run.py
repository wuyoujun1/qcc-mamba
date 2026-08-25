import os
try:
    import setproctitle
    setproctitle.setproctitle(os.environ.get('QCC_HIDE_NAME', 'dataops_worker'))
except Exception:
    pass

import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')
    # seed 参数化（阶段二 3-seed 确认用；默认 2023 保持与探索阶段一致）
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='S_Mamba',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,S_Mamba ]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')
    # Frequency-Token S-Mamba (Q_S_Mamba_ft)：把对齐频谱投影成 token 拼进反向嵌入
    parser.add_argument('--freq_tokens', type=int, default=0, help='0=off, 1=backbone frequency tokens, 2=freq tokens + quantum mix (Q_S_Mamba_ft)')
    # FreDF 频域监督（纯训练项，零结构改动）
    parser.add_argument('--freq_loss', type=int, default=0, help='1 = FreDF 频域监督损失（时域 MSE + λ·频域 MAE）')
    parser.add_argument('--freq_lambda', type=float, default=0.5, help='FreDF 频域损失权重 λ')
    parser.add_argument('--freq_lowpass', type=float, default=1.0, help='FreDF 低频加权：只对最低比例的低频 bin 计频域损失（长 horizon 防高频噪声惩罚）')
    parser.add_argument('--freq_mode', type=str, default='complex', help='FreDF 频域项：complex=原始 FreDF（复数差幅值）；amp_phase=振幅/相位分离监督（频域双轴对齐强化，对强周期数据更有效）')
    # S_Mamba_freqline：FITS 风格频率主线旁路（轻量架构增强，主干不变）
    parser.add_argument('--freq_mainline', type=int, default=0, help='1 = S_Mamba_freqline：对输入 rFFT→低频滤波→iFFT 叠加周期主线到输出（模型= S_Mamba_freqline）')
    parser.add_argument('--freq_mainline_scale', type=float, default=1.0, help='频率主线叠加缩放')
    parser.add_argument('--freq_lowpass_mainline', type=float, default=0.2, help='频率主线低通比例（保留最低比例 bin）')
    # S_Mamba_freqproj：频域基投影（改造 projector，主干不变）
    parser.add_argument('--freqproj', type=int, default=0, help='1 = S_Mamba_freqproj：DCT 基重构预测（模型= S_Mamba_freqproj）')
    parser.add_argument('--freqproj_k', type=int, default=0, help='DCT 系数个数（0 = pred_len 全频；小 K = 低通约束）')
    parser.add_argument('--freqproj_learnable', type=int, default=1, help='1 = 允许微调 DCT 基')
    parser.add_argument('--amp_w', type=float, default=1.0, help='amp_phase 模式下振幅项权重')
    parser.add_argument('--phase_w', type=float, default=1.0, help='amp_phase 模式下相位项权重')
    # Q-S-Mamba（量子混合，qcc 移植）-- qmix_layers=0 保持官方 S-Mamba 行为
    parser.add_argument('--qmix_layers', type=int, default=0, help='quantum mix layers after encoder layers (0 = official S-Mamba)')
    parser.add_argument('--n_qubits', type=int, default=2, help='number of qubits N (state dim 2^N)')
    parser.add_argument('--qmix_n_layers', type=int, default=2, help='data-reupload layers D of quantum feature map')
    parser.add_argument('--qmix_norm', type=str, default='avg', help='message-passing norm: avg | softmax | l1')
    parser.add_argument('--kernel_T', type=float, default=1.0, help='fidelity kernel temperature T (softmax(K/T))')
    parser.add_argument('--offdiag', action='store_true', help='softmax on (K - I) off-diagonal weights')
    parser.add_argument('--topk', type=int, default=0, help='top-k couplings per row renormalized (0 = off)')
    parser.add_argument('--entangle_topo', type=str, default='linear', help='entanglement topology: linear | ring | none')
    parser.add_argument('--kernel_fn', type=str, default='quantum', help='kernel: quantum | quantum_exp | quantum_power | rbf | periodic | rff | linear_imag | linear_real | none')
    parser.add_argument('--kernel_exp', type=float, default=1.0, help='quantum_exp 带宽 κ（Bures 角测地核的指数系数）')
    parser.add_argument('--kernel_power', type=float, default=2.0, help='quantum_power 锐化指数 p（K=f^p）')
    parser.add_argument('--kernel_exp_learn', action='store_true', help='quantum_exp 带宽 κ 可学习（自调带宽，rbf 固定 γ 做不到）')
    parser.add_argument('--kernel_power_learn', action='store_true', help='quantum_power 指数 p 可学习')
    parser.add_argument('--kernel_phase', type=float, default=1.0, help='quantum_phase_exp 相位强度 λ（配 qmix_norm=l1 保留方向）')
    parser.add_argument('--qmix_use_fmap', type=int, default=1, help='1 = 核吃量子态 ψ（feature map）；0 = 核直接吃原始 H（经典对照）')
    parser.add_argument('--align_lambda', type=float, default=0.0, help='kernel alignment：让 K 对齐预测目标跨变量相关结构（>0 开启，参数化量子核的差异化能力）')
    parser.add_argument('--align_mode', type=str, default='time', help='kernel alignment 目标：time=目标时域相关 | freq=目标频域跨谱相干(低通,接 FreDF) | both')
    parser.add_argument('--align_norm', type=int, default=0, help='1 = 对齐归一化消息权重 K_n（softmax/offdiag 后下游真正用的权重），而非原始 K')
    parser.add_argument('--align_lowpass', type=float, default=0.5, help='freq 对齐目标低通比例（保留最低比例 bin，与 FreDF freq_lowpass 同思路）')
    parser.add_argument('--save_pred', type=int, default=1, help='0 = 不保存 pred.npy/true.npy（大变量数据集如 ECL/traffic 防止占满根盘）')
    parser.add_argument('--mem_safe', type=int, default=0, help='1 = 测试用增量 MSE/MAE（避免 traffic:720 等大张量堆叠 OOM，14G 数组问题）')
    parser.add_argument('--qmix_joint', action='store_true', help='JEQK：跨变量纠缠联合编码（V 变量 → V·nq qubit 纠缠态，核用约化密度矩阵）')
    parser.add_argument('--joint_topo', type=str, default='var_linear', help='联合纠缠拓扑: var_linear | var_ring | none')
    parser.add_argument('--angle_norm', type=str, default='clamp', help='angle normalization: clamp | sphere')
    parser.add_argument('--theta_S_scale0', type=float, default=0.5, help='initial S-modulation strength gamma')
    parser.add_argument('--qmix_gate', action='store_true', help='learnable gate (gamma=0 -> output==input)')
    parser.add_argument('--qmix_use_H', type=int, default=1, help='K reads H (1) or spectrum S only (0, P1-1)')
    parser.add_argument('--qmix_fixed_s_scale', action='store_true', help='P1-1b: S enters fmap at fixed scale (un-suppressible)')
    parser.add_argument('--qmix_gate_init', type=float, default=0.0, help='initial gate value')
    parser.add_argument('--qmix_gate_pv', action='store_true', help='P2: per-variable gate (each target var learns whether to accept cross-var message; adapt to large-V weak-coupling datasets)')
    parser.add_argument('--qmix_gate_pv_init', type=float, default=3.0, help='per-variable gate logit init (sigmoid); 3.0≈open(0.95), -3.0≈closed(0.05)')
    parser.add_argument('--qmix_gate_pv_src', action='store_true', help='P2b: per-source gate = learnable topk (each source var w learns whether to participate in cross-var messages; K column selectivity)')
    parser.add_argument('--qmix_gate_pv_src_init', type=float, default=-3.0, help='per-source gate logit init (sigmoid); -3.0≈closed(0.05)')
    parser.add_argument('--spectrum_M', type=int, default=32, help='spectrum resample points M (S dim = 2M, or 2M+1 with delay_in_s)')
    parser.add_argument('--spectrum_time_align', action='store_true', help='time-axis alignment via FFT cross-correlation')
    parser.add_argument('--spectrum_freq_align', action='store_true', help='frequency-axis alignment via peak-frequency resample')
    parser.add_argument('--spectrum_range', type=str, default='0_2', help='resample range: 0_2 | 0_1')
    parser.add_argument('--spectrum_amp_normalize', action='store_true', help='amplitude normalization A/A_max')
    parser.add_argument('--delay_in_s', action='store_true', help='append time-shift delta_hat to S (S dim 2M+1)')
    parser.add_argument('--hp_scale_v', action='store_true', help='hp_scale = 7/V (auto-adapt to variable count)')
    parser.add_argument('--qmix_use_S_only', action='store_true', help='P1-1: quantum kernel reads S only (not H)')
    parser.add_argument('--qmix_amplitude_encoding', action='store_true', help='第四轮：振幅编码（零信息损失）')
    # 第五轮：多组角度桥接（不用振幅编码）
    parser.add_argument('--angle_groups', type=int, default=1, help='多组角度桥接：把 2M 维 S 拆 G 组，每组独立 2N 角度')
    parser.add_argument('--kernel_group_agg', type=str, default='product', help='多组核聚合: product | mean')

    # P2-1 双路径（2026-08-17）：时间 SSM 单向 + 量子核独占跨变量
    parser.add_argument('--dp_time_layers', type=int, default=2, help='time path SSM layers')
    parser.add_argument('--dp_time_dim', type=int, default=256, help='time path hidden dim')
    parser.add_argument('--dp_time_pool', type=str, default='mean', help='time pool: mean | last')
    parser.add_argument('--dp_var_embed', type=int, default=1, help='per-variable identity embedding')
    parser.add_argument('--dp_msg', type=str, default='S', help='var-path message source: S | H | both')
    parser.add_argument('--dp_fusion', type=str, default='add', help='fusion: add | time_only (new plain baseline)')
    parser.add_argument('--dp_gate_init', type=float, default=0.05, help='fusion gate init (gamma=0 -> H == H_time)')
    parser.add_argument('--use_dp_feats', action='store_true', help='broadcast time features into TimePath')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.is_training==1:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        print(11111)
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)
            exp = Exp(args)
            exp.get_input(setting)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
