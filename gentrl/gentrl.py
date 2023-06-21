import torch
import torch.nn as nn
import torch.optim as optim
from math import pi, log
from gentrl.lp import LP
import pickle
import logging
import os
import numpy as np
import selfies as sf
# import selfies as sf

from moses.metrics.utils import get_mol

import warnings
warnings.filterwarnings("ignore")


class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def print(self):
        for key in self.stats.keys():
            print(str(key) + ": {:4.4};".format(
                sum(self.stats[key]) / len(self.stats[key])
            ), end='')

        print()


class GENTRL(nn.Module):
    '''
    GENTRL model
    '''
    def __init__(self, enc, dec, latent_descr, feature_descr, tt_int=40,
                 tt_type='usual', beta=0.01, gamma=0.1):
        super(GENTRL, self).__init__()

        self.enc = enc
        self.dec = dec

        self.num_latent = len(latent_descr)
        self.num_features = len(feature_descr)

        self.latent_descr = latent_descr
        self.feature_descr = feature_descr

        self.tt_int = tt_int  # m
        self.tt_type = tt_type

        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type)

        self.beta = beta
        self.gamma = gamma
        logging.info("finish model initialization")

    def get_elbo(self, x, y):
        means, log_stds = torch.split(self.enc.encode(x),
                                      len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = self.dec.weighted_forward(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        zy = torch.cat([latvar_samples, y], dim=1)
        log_p_zy = self.lp.log_prob(zy)

        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)
        log_p_z_by_y = log_p_zy - log_p_y
        log_p_y_by_z = log_p_zy - log_p_z

        kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()

        elbo = rec_part - self.beta * kldiv_part
        elbo = elbo + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy(),
            'log_p_y_by_z': log_p_y_by_z.mean().detach().cpu().numpy(),
            'log_p_z_by_y': log_p_z_by_y.mean().detach().cpu().numpy()
        }

    def save(self, folder_to_save='./'):
        if folder_to_save[-1] != '/':
            folder_to_save = folder_to_save + '/'
        torch.save(self.enc.state_dict(), folder_to_save + 'enc.model')
        torch.save(self.dec.state_dict(), folder_to_save + 'dec.model')
        torch.save(self.lp.state_dict(), folder_to_save + 'lp.model')

        pickle.dump(self.lp.order, open(folder_to_save + 'order.pkl', 'wb'))

        logging.info("model saved")

    def load(self, folder_to_load='./'):
        if folder_to_load[-1] != '/':
            folder_to_load = folder_to_load + '/'

        order = pickle.load(open(folder_to_load + 'order.pkl', 'rb'))
        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type,
                     order=order)

        self.enc.load_state_dict(torch.load(folder_to_load + 'enc.model'))
        self.dec.load_state_dict(torch.load(folder_to_load + 'dec.model'))
        self.lp.load_state_dict(torch.load(folder_to_load + 'lp.model'))

        logging.info("model loaded")

    def train_as_vaelp(self, train_loader, num_epochs=1000,
                       verbose_step=50, lr=1e-3,file_path = '',dec_ratio=0.1):
        lr_dec = lr * dec_ratio
        print("current_dec_lr is:"+str(lr_dec))
        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
        
#         os.mkdir(file_path)
        global_stats = TrainStats()
        local_stats = TrainStats()
        loss_lis = []
        stats_dic = []    
        epoch_i = 0
        to_reinit = False
        buf = None
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.95)
        scheduler_dec = optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda = lambda i: 0.95**i)
#         scheduler_dec = optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda = lambda i: 0 if i<10 else 0.95**(i-10))
        while epoch_i < num_epochs:
            i = 0
            if verbose_step:
                print("Epoch", epoch_i, ":")

            if epoch_i in [0, 1, 5, 10, 15,20, 25,30, 35,40, 50,60,70,80,90]:
                to_reinit = True

            epoch_i += 1

            for x_batch, y_batch in train_loader:
                if verbose_step:
                    print("!", end='')

                i += 1

                y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.view(-1, 1).contiguous()

                if to_reinit:
                    if (buf is None) or (buf.shape[0] < 5000):
                        enc_out = self.enc.encode(x_batch)
                        means, log_stds = torch.split(enc_out,
                                                      len(self.latent_descr),
                                                      dim=1)
                        z_batch = (means + torch.randn_like(log_stds) *
                                   torch.exp(0.5 * log_stds))
                        cur_batch = torch.cat([z_batch, y_batch], dim=1)
                        if buf is None:
                            buf = cur_batch
                        else:
                            buf = torch.cat([buf, cur_batch])
                    else:
                        descr = len(self.latent_descr) * [0]
                        descr += len(self.feature_descr) * [1]
                        self.lp.reinit_from_data(buf, descr)
                        self.lp.cuda()
                        buf = None
                        to_reinit = False

                    continue

                elbo, cur_stats = self.get_elbo(x_batch, y_batch)
                loss_lis.append(cur_stats['loss'])
                local_stats.update(cur_stats)
                global_stats.update(cur_stats)

                optimizer.zero_grad()
                optimizer_dec.zero_grad()
                loss = -elbo
                loss.backward()
                optimizer.step()
                
                
                
#                 batch_size=len(x_batch)
#                 z = self.lp.sample(batch_size, 50 * ['s'] + ['m']) 

#                 smiles = self.dec.sample(50, z, argmax=False)
#                 log_probs = self.dec.weighted_forward(smiles, z)
#                 r_list = [1 if get_mol(s) is not None else 0 for s in smiles]
#                 rewards = torch.tensor(r_list).float().to(z.device)
#                 rewards_bl = rewards - rewards.mean()
                
#                 loss = -(log_probs * rewards_bl).mean()
#                 loss.backward()
                                
#                 optimizer_dec.zero_grad()
#                 optimizer_dec.step()
                
                optimizer_dec.step()
                
#                 print("第%d个epoch的学习率：%f" % (epoch_i, optimizer.param_groups[0]['lr']))
                if verbose_step and i % verbose_step == 0:
                    local_stats.print()
                    stats_dic.append(local_stats)
                    local_stats.reset()
                    i = 0
               

            if epoch_i % 5==0:
                p = "models_epoch_"+str(epoch_i)
                os.mkdir(file_path+'/'+p)
                self.save(file_path+'/'+p)
                print(p+"   saved")
            scheduler.step()
            scheduler_dec.step()
            print("第%d个epoch的学习率：%f" % (epoch_i, optimizer.param_groups[0]['lr']))
            print("第%d个epoch的dec学习率：%f" % (epoch_i, optimizer_dec.param_groups[0]['lr']))
            if i > 0:
                local_stats.print()
                stats_dic.append(local_stats)
                local_stats.reset()

        return global_stats, local_stats, loss_lis

    def train_as_rl(self,
                    reward_fn,
                    num_iterations=1500, verbose_step=50,
                    batch_size=100,
                    cond_lb=-2, cond_rb=0,
                    lr_lp=1e-3, lr_dec=1e-4,exploration_ratio = 0.1,file_path='',topN=1):
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)

        global_stats = TrainStats()
        local_stats = TrainStats()
        record_mean_reward = []
        record_valid_perc = []
#         record_loss = []
#         record_mean_som = []
        record_reward_bl = []
        record_mean_mfp_sum = []
        record_mean_bayes = []
        cur_iteration = 0
        while cur_iteration < num_iterations:
#             print("cur_iteration:"+str(cur_iteration),end='')
            print("!", end='')
            
            exploit_size = int(batch_size * (1 - exploration_ratio))
            exploit_z = self.lp.sample(exploit_size, 50 * ['s'] + ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            expl_size = int(batch_size * exploration_ratio)
            expl_z = torch.randn(expl_size, exploit_z.shape[1])
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            z = torch.cat([exploit_z, expl_z])
#             print("*",end='')
            smiles = self.dec.sample(50, z, argmax=False)
            
            
            # add distinct rf
#             print('len smiles:',len(smiles))
            sf_smiles = [sf.decoder(s) for s in smiles]
            r_list = reward_fn(sf_smiles)
#             r_list = np.array([list(reward_fn(s,cur_iteration)) for s in smiles])
#             r_list = r_list.sum(
            # add feature of topN
            
#             r_list = np.array([list(reward_fn(s)) for s in smiles])
            rewards = r_list.sum(axis=1)
            split_pos = int(topN * len(rewards))
            pos = np.argpartition(rewards,-split_pos)[-split_pos:]
            rewards = rewards[pos]
            
            rewards = torch.tensor(rewards).float().to(exploit_z.device)
            
            z = z[pos]
            smiles = list(np.array(smiles)[pos])
            zc = torch.zeros(z.shape[0], 1).to(z.device)
            conc_zy = torch.cat([z, zc], dim=1)
            log_probs = self.lp.log_prob(conc_zy, marg=50 * [False] + [True])
            log_probs += self.dec.weighted_forward(smiles, z)
            
            
            rewards_bl = rewards - rewards.mean()
            mean_reward_bl = rewards_bl.mean()

            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()
 
            valid_sm = [s for s in sf_smiles if get_mol(s) is not None]
#             mean_reward = rewards.mean()
#             mean_bayes = r_list[:,2:].sum(axis=0) / len(smiles)
#             mean_mfp_sum = np.nanmean(r_list[:,1])#.nanmean()
            
            mean_reward = r_list.sum() / len(smiles)
#             mean_som = r_list.sum(axis=0) / len(smiles)
            valid_perc = len(valid_sm) / len(smiles)
            cur_stats = {'r_list_sum':r_list.sum(),
                         'mean_reward': mean_reward,
                         'valid_perc': valid_perc}

            local_stats.update(cur_stats)
            global_stats.update(cur_stats)
            record_mean_reward.append(mean_reward)
            record_valid_perc.append(valid_perc)
#             record_mean_som.append(mean_som)
#             record_mean_bayes.append(mean_bayes)
#             record_mean_mfp_sum.append(mean_mfp_sum)
            
#             record_loss.append(loss)
#             record_reward_bl.append(mean_reward_bl)
            cur_iteration += 1

            if verbose_step and (cur_iteration + 1) % verbose_step == 0:

                local_stats.print()
                local_stats.reset()
                print("\ncur_iteration:"+str(cur_iteration)+"\n")
            if ((cur_iteration+1)%1000==0) or (400<=cur_iteration<1000 and (cur_iteration+1)%100==0)or(cur_iteration<400  and (cur_iteration+1)%10==0):
                path = file_path+"/SOM_model_iter"+str(cur_iteration)
                os.mkdir(path)
                self.save("./"+path)
                print("\nmodel_iter"+str(cur_iteration)+" saved\n")

        return global_stats, record_mean_reward, record_valid_perc#,record_mean_bayes,record_mean_mfp_sum
    
#     def increase_vaelp_validity(self, train_loader, num_epochs=10,
#                        verbose_step=50, lr=1e-3):
#         optimizer = optim.Adam(self.parameters(), lr=lr)
#         optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr)

#         global_stats = TrainStats()
#         local_stats = TrainStats()

#         epoch_i = 0
#         to_reinit = False
#         buf = None


#         i = 0
#         if verbose_step:
#             print("Epoch", epoch_i, ":")

#         if epoch_i in [0, 1, 5]:
#             to_reinit = True

#         epoch_i += 1

#         if epoch_i % 2 == 0:
#             for x_batch, y_batch in train_loader:
#                 if verbose_step:
#                     print("!", end='')

#                 i += 1

#                 y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
#                 if len(y_batch.shape) == 1:
#                     y_batch = y_batch.view(-1, 1).contiguous()

#                 if to_reinit:
#                     if (buf is None) or (buf.shape[0] < 5000):
#                         enc_out = self.enc.encode(x_batch)
#                         means, log_stds = torch.split(enc_out,
#                                                       len(self.latent_descr),
#                                                       dim=1)
#                         z_batch = (means + torch.randn_like(log_stds) *
#                                    torch.exp(0.5 * log_stds))
#                         cur_batch = torch.cat([z_batch, y_batch], dim=1)
#                         if buf is None:
#                             buf = cur_batch
#                         else:
#                             buf = torch.cat([buf, cur_batch])
#                     else:
#                         descr = len(self.latent_descr) * [0]
#                         descr += len(self.feature_descr) * [1]
#                         self.lp.reinit_from_data(buf, descr)
#                         self.lp.cuda()
#                         buf = None
#                         to_reinit = False

#                     continue

#                 elbo, cur_stats = self.get_elbo(x_batch, y_batch)
#                 local_stats.update(cur_stats)
#                 global_stats.update(cur_stats)

#                 optimizer.zero_grad()
#                 loss = -elbo
#                 loss.backward()
#                 optimizer.step()
                
                
#                 batch_size=len(x_batch)
#                 z = self.lp.sample(batch_size, 50 * ['s'] + ['m']) 

#                 smiles = self.dec.sample(50, z, argmax=False)
#                 log_probs = self.dec.weighted_forward(smiles, z)
#                 r_list = [1 if get_mol(s) is not None else 0 for s in smiles]
#                 rewards = torch.tensor(r_list).float().to(z.device)
#                 rewards_bl = rewards - rewards.mean()
                
#                 loss = -(log_probs * rewards_bl).mean()
#                 loss.backward()
                                
#                 optimizer_dec.zero_grad()
#                 optimizer_dec.step()
                
#                 if verbose_step and i % verbose_step == 0:
#                     local_stats.print()
#                     local_stats.reset()
#                     i = 0

#             epoch_i += 1
#             if i > 0:
#                 local_stats.print()
#                 local_stats.reset()

#         return global_stats
    
    def increase_vaelp_validity(self, train_loader, num_epochs=1000,
                       verbose_step=50, lr=1e-3,file_path = '',dec_ratio=0.5):
        lr_dec = lr * dec_ratio
        print("current_dec_lr is:"+str(lr_dec))
        optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
        
#         os.mkdir(file_path)
        global_stats = TrainStats()
        local_stats = TrainStats()
        loss_lis = []
        stats_dic = []    
        epoch_i = 0
        to_reinit = False
        buf = None
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.95)
        scheduler_dec = optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda = lambda i: 0.95**i)
#         scheduler_dec = optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda = lambda i: 0 if i<10 else 0.95**(i-10))
        while epoch_i < num_epochs:
            i = 0
            if verbose_step:
                print("Epoch", epoch_i, ":")

            if epoch_i in [0, 1, 5, 10, 15,20, 25,30, 35,40, 50,60,70,80,90]:
                to_reinit = True

            epoch_i += 1

            for x_batch, y_batch in train_loader:
                if verbose_step:
                    print("!", end='')

                i += 1

                y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.view(-1, 1).contiguous()

                if to_reinit:
                    if (buf is None) or (buf.shape[0] < 5000):
                        enc_out = self.enc.encode(x_batch)
                        means, log_stds = torch.split(enc_out,
                                                      len(self.latent_descr),
                                                      dim=1)
                        z_batch = (means + torch.randn_like(log_stds) *
                                   torch.exp(0.5 * log_stds))
                        cur_batch = torch.cat([z_batch, y_batch], dim=1)
                        if buf is None:
                            buf = cur_batch
                        else:
                            buf = torch.cat([buf, cur_batch])
                    else:
                        descr = len(self.latent_descr) * [0]
                        descr += len(self.feature_descr) * [1]
                        self.lp.reinit_from_data(buf, descr)
                        self.lp.cuda()
                        buf = None
                        to_reinit = False

                    continue

                elbo, cur_stats = self.get_elbo(x_batch, y_batch)
                loss_lis.append(cur_stats['loss'])
                local_stats.update(cur_stats)
                global_stats.update(cur_stats)

                optimizer.zero_grad()
#                 optimizer_dec.zero_grad()
                loss = -elbo
                loss.backward()
                optimizer.step()
                
                
                
                batch_size=len(x_batch)
#                 smiles = self.sample(batch_size)
                z = self.lp.sample(batch_size, 50 * ['s'] + ['m']) 
                smiles = self.dec.sample(100, z, argmax=False)
                log_probs = self.dec.weighted_forward(smiles, z)
#                 r_list = list(map(get_mol,smiles))
                r_list = [1 if get_mol(s) is not None else 0 for s in smiles]
                rewards = torch.tensor(r_list).float().to(z.device)
                rewards_bl = rewards - rewards.mean()
                
                optimizer_dec.zero_grad()
                loss = -(log_probs * rewards_bl).mean()
                loss.backward()
          
                optimizer_dec.step()
                
                
#                 print("第%d个epoch的学习率：%f" % (epoch_i, optimizer.param_groups[0]['lr']))
                if verbose_step and i % verbose_step == 0:
                    local_stats.print()
                    stats_dic.append(local_stats)
                    local_stats.reset()
                    i = 0
               

            if epoch_i % 5==0:
                p = "models_epoch_"+str(epoch_i)
                os.mkdir(file_path+'/'+p)
                self.save(file_path+'/'+p)
                print(p+"   saved")
            scheduler.step()
            scheduler_dec.step()
            print("第%d个epoch的学习率：%f" % (epoch_i, optimizer.param_groups[0]['lr']))
            print("第%d个epoch的dec学习率：%f" % (epoch_i, optimizer_dec.param_groups[0]['lr']))
            if i > 0:
                local_stats.print()
                stats_dic.append(local_stats)
                local_stats.reset()

        return global_stats, local_stats, loss_lis

    def sample(self, num_samples):
        z = self.lp.sample(num_samples, 50 * ['s'] + ['m'])
        smiles = self.dec.sample(100, z, argmax=False)

        return smiles
