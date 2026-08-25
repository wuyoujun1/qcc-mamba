import random

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np


class FreDFLoss(nn.Module):
    """FreDF 频域监督损失（ICLR'24）：时域 MSE + λ·频域项（rFFT 上）。

    freq_lowpass (0,1]：只对最低比例的低频 bin 计频域损失。
      长 horizon（如 720）时全频带等权会过度惩罚高频噪声 bin（实测 720 反而变差 +7%），
      低频加权（借鉴 FITS 低通思想）聚焦有信息量的低频结构。

    freq_mode（频域双轴对齐强化）：
      'complex'（默认，即原始 FreDF）：L_f = |FFT(pred) - FFT(true)|（复数差幅值）
      'amp_phase'：L_f = α·|Amp差| + β·|Phase差|。分离振幅/相位监督——
        振幅项聚焦周期性强度，相位项聚焦周期相位对齐（对 ECL 这类强周期数据更有效）。
    """

    def __init__(self, freq_lambda: float = 0.5, freq_lowpass: float = 1.0,
                 freq_mode: str = 'complex', amp_w: float = 1.0, phase_w: float = 1.0):
        super().__init__()
        self.freq_lambda = freq_lambda
        self.freq_lowpass = freq_lowpass
        self.freq_mode = freq_mode
        self.amp_w = amp_w
        self.phase_w = phase_w

    def forward(self, pred, true):
        l_t = nn.functional.mse_loss(pred, true)
        # rFFT 沿时间轴（pred_len）；复数差（实部+虚部）
        pred_f = torch.fft.rfft(pred, dim=1)
        true_f = torch.fft.rfft(true, dim=1)
        if self.freq_lowpass < 1.0:
            n_bins = pred_f.shape[1]
            keep = max(1, int(n_bins * self.freq_lowpass))
            pred_f = pred_f[:, :keep, :]
            true_f = true_f[:, :keep, :]
        if self.freq_mode == 'amp_phase':
            # 振幅/相位分离（频域双轴对齐强化）
            amp_pred, ph_pred = torch.abs(pred_f), torch.angle(pred_f)
            amp_true, ph_true = torch.abs(true_f), torch.angle(true_f)
            # 相位差用角距离（环上 wrap），避免 0/2π 突变
            phase_diff = torch.atan2(torch.sin(ph_pred - ph_true), torch.cos(ph_pred - ph_true))
            l_f = self.amp_w * torch.mean(torch.abs(amp_pred - amp_true)) \
                  + self.phase_w * torch.mean(torch.abs(phase_diff))
        else:
            diff = pred_f - true_f
            l_f = torch.mean(torch.abs(diff))
        return l_t + self.freq_lambda * l_f

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # FreDF（ICLR'24）：频域监督弥补直接预测在标签自相关下的估计偏差。
        # L = MSE_t + λ·|FFT(pred) - FFT(true)|（频域 MAE）。零结构改动，纯训练项。
        if getattr(self.args, "freq_loss", 0):
            return FreDFLoss(
                freq_lambda=getattr(self.args, "freq_lambda", 0.5),
                freq_lowpass=getattr(self.args, "freq_lowpass", 1.0),
                freq_mode=getattr(self.args, "freq_mode", 'complex'),
                amp_w=getattr(self.args, "amp_w", 1.0),
                phase_w=getattr(self.args, "phase_w", 1.0),
            )
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # kernel alignment（QF-1, 2026-08-24）：让量子核耦合矩阵对齐预测目标的跨变量耦合结构。
                # 三个修复 vs 原版：
                #  1. 对齐目标可选 freq = 目标频域跨谱相干（低通，接 FreDF 频谱监督）——"量子核适配 FreDF"；
                #  2. align_norm=1 时对齐归一化消息权重 K_n（softmax/offdiag 后下游真正用的权重），
                #     而非原始 K —— 修复"对齐塑形被 softmax 洗掉"的根因；
                #  3. align_lambda 建议从 0.1 起步（曾 1.0 让量子核崩）。
                align_lambda = getattr(self.args, 'align_lambda', 0.0)
                if align_lambda > 0:
                    align_mode = getattr(self.args, 'align_mode', 'time')
                    align_norm = int(getattr(self.args, 'align_norm', 0))
                    align_lowpass = float(getattr(self.args, 'align_lowpass', 0.5))
                    K = (getattr(self.model, '_last_K_norm', None) if align_norm
                         else getattr(self.model, '_last_K', None))
                    if K is not None and K.shape[0] == batch_y.shape[0] and K.shape[1] == batch_y.shape[2]:
                        Vk = K.shape[-1]
                        C = None
                        if align_mode in ('time', 'both'):
                            # 目标：时域跨变量相关 C[b,i,j] = corr(y_i, y_j)
                            ym = batch_y - batch_y.mean(1, keepdim=True)
                            cov = torch.einsum('bpi,bpj->bij', ym, ym) / max(batch_y.shape[1] - 1, 1)
                            sd = cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
                            C = cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
                        if align_mode in ('freq', 'both'):
                            # 目标：频域跨谱相干（低通）—— Σ_f Re[Y_i(f)·conj(Y_j(f))] / √(P_i·P_j)
                            Y = torch.fft.rfft(batch_y, dim=1)  # (B, nbins, V)
                            nb = Y.shape[1]
                            keep = max(1, int(nb * align_lowpass))
                            Yl = Y[:, :keep]
                            CSD = (Yl.real.transpose(-1, -2) @ Yl.real) + (Yl.imag.transpose(-1, -2) @ Yl.imag)  # (B,V,nbins)@(B,nbins,V)->(B,V,V)
                            sdf = CSD.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
                            C_f = CSD / (sdf.unsqueeze(-1) * sdf.unsqueeze(-2))
                            C = C_f if C is None else 0.5 * C + 0.5 * C_f
                        assert C.shape[-1] == Vk, f"align target shape mismatch: C={C.shape} K={K.shape}"
                        # 去对角（K/C 对角恒 1，中心化后对角主导余弦，对齐信号被稀释）
                        eye_mask = torch.eye(Vk, device=K.device).unsqueeze(0)
                        Kc = K * (1 - eye_mask)
                        Cc = C * (1 - eye_mask)
                        Kcs = Kc - Kc.mean(dim=(-2, -1), keepdim=True)
                        Ccs = Cc - Cc.mean(dim=(-2, -1), keepdim=True)
                        align = (Kcs * Ccs).sum(dim=(-2, -1)) / (
                            (Kcs.norm(dim=(-2, -1)) * Ccs.norm(dim=(-2, -1))).clamp_min(1e-8)
                        )
                        loss = loss - align_lambda * align.mean()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    # print(speed)
                    # allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                    # cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                    # total = allocated_memory + cached_memory
                    # print('allocated_memory:', allocated_memory)
                    # print('cached_memory:', cached_memory)
                    # print('total:', total)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mem_safe = bool(getattr(self.args, 'mem_safe', 0))
        mse_acc = 0.0
        mae_acc = 0.0
        cnt = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # time_points = random.sample(range(batch_x.size()[1]), 5)
                # 假设您有一个包含每个变量标准差的张量stds，形状为(321,)
                # 定义扰动强度
                # epsilon = 1
                # # 创建一个与原始张量形状相同的张量来存储扰动
                # perturbed_tensor = torch.zeros_like(batch_x)
                # # 对每个选定的时间点添加扰动
                # for time_point in time_points:
                #     # 生成与tensor在该时间点形状相同的随机噪声
                #     noise = torch.randn(1, 321) * epsilon
                #     perturbed_tensor[:, time_point, :] += noise.float().to(self.device)
                # batch_x += perturbed_tensor
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if mem_safe:
                    mse_acc += float(((pred - true) ** 2).sum())
                    mae_acc += float(np.abs(pred - true).sum())
                    cnt += pred.size
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if mem_safe:
            mse = mse_acc / max(cnt, 1)
            mae = mae_acc / max(cnt, 1)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, mse, mse, mse]))
            # 跳过 preds/trues 大数组堆叠与保存（内存安全模式）
            if not getattr(self.args, 'save_pred', 1):
                return mse
        else:
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if getattr(self.args, 'save_pred', 1):
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return