import logging
import os
import time
import torch
import pykp.utils.io as io
from pykp.utils.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.string_helper import *
from utils.functions import time_since
from pykp.utils.label_assign import hungarian_assign

EPS = 1e-8
def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, = batch

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            n_batch += batch_size
            word2idx = opt.vocab['word2idx']
            target = trg_oov if opt.copy_attention else trg

            start_time = time.time()
            if opt.fix_kp_num_len:
                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                control_embed = model.decoder.forward_seg(state)

                y_t_init = target.new_ones(batch_size, opt.max_kp_num, 1) * word2idx[io.BOS_WORD]
                if opt.set_loss:  # reassign target
                    input_tokens = src.new_zeros(batch_size, opt.max_kp_num, opt.assign_steps + 1)
                    decoder_dists = []
                    input_tokens[:, :, 0] = word2idx[io.BOS_WORD]
                    for t in range(1, opt.assign_steps + 1):
                        decoder_inputs = input_tokens[:, :, :t]
                        decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(opt.vocab_size - 1),
                                                                    word2idx[io.UNK_WORD])

                        decoder_dist, _ = model.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
                        input_tokens[:, :, t] = decoder_dist.argmax(-1)
                        decoder_dists.append(decoder_dist.reshape(batch_size, opt.max_kp_num, 1, -1))
                    decoder_dists = torch.cat(decoder_dists, -2)

                    
                    # second stage generation with token null masked
                    state = model.decoder.init_state(memory_bank, src_mask)
                    control_embed = model.decoder.forward_seg(state)
                    input_tokens_masknull = src.new_zeros(batch_size, opt.max_kp_num, opt.assign_steps + 1)
                    input_tokens_masknull[:, :, 0] = word2idx[io.BOS_WORD]
                    for t in range(1, opt.assign_steps + 1):
                        decoder_inputs_masknull = input_tokens_masknull[:, :, :t]
                        decoder_inputs_masknull = decoder_inputs_masknull.\
                                        masked_fill(decoder_inputs_masknull.gt(opt.vocab_size - 1),
                                                                    word2idx[io.UNK_WORD])

                        decoder_dist_masknull, _ = model.decoder(decoder_inputs_masknull,\
                                                   state, src_oov, max_num_oov, control_embed)
                        decoder_dist_masknull[:, :, word2idx[io.NULL_WORD]] = 0
                        input_tokens_masknull[:, :, t] = decoder_dist_masknull.argmax(-1)

                    
                    
                    if opt.seperate_pre_ab:
                        mid_idx = opt.max_kp_num // 2
                        pre_reorder_index = hungarian_assign(decoder_dists[:, :mid_idx],
                                                             target[:, :mid_idx, :opt.assign_steps],
                                                             ignore_indices=[word2idx[io.NULL_WORD],
                                                                             word2idx[io.PAD_WORD]])
                        target[:, :mid_idx] = target[:, :mid_idx][pre_reorder_index]
                        trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                        ab_reorder_index = hungarian_assign(decoder_dists[:, mid_idx:],
                                                            target[:, mid_idx:, :opt.assign_steps],
                                                            ignore_indices=[word2idx[io.NULL_WORD],
                                                                            word2idx[io.PAD_WORD]])
                        target[:, mid_idx:] = target[:, mid_idx:][ab_reorder_index]
                        trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]
                    else:
                        reorder_index = hungarian_assign(decoder_dists, target[:, :, :opt.assign_steps],
                                                         [word2idx[io.NULL_WORD],
                                                          word2idx[io.PAD_WORD]])
                        target = target[reorder_index]
                        trg_mask = trg_mask[reorder_index]

            
            # select the correctly predicted slots
            masknull_tokensum = input_tokens_masknull[:, :, 1:].sum(-1).unsqueeze(-1)
            masknull_tokensum = masknull_tokensum.repeat((1, 1, opt.max_kp_num))
            target_tokensum = target[:, :, :opt.assign_steps].sum(-1).unsqueeze(1)
            target_tokensum = target_tokensum.repeat((1, opt.max_kp_num, 1))

            correct_slots = (masknull_tokensum == target_tokensum).sum(-1).float()
            keyphrase_mask = (target[:, :, 0] == word2idx[io.NULL_WORD]).float()

            
            keyphrase_tokensum = input_tokens[:, :, 1:].sum(-1).unsqueeze(-1)
            keyphrase_tokensum = keyphrase_tokensum.repeat((1, 1, opt.max_kp_num))
            keyphrase_tokensum = torch.einsum('bnj,bn->bnj', masknull_tokensum, 1-keyphrase_mask)
            keyphrase_tokensum = torch.einsum('bjn,bn->bjn', keyphrase_tokensum, 1-keyphrase_mask)
            key_correct_slots = (keyphrase_tokensum == target_tokensum).sum(-1).float()
    
            
            correct_null_slots = 1 - torch.einsum('bn,bn->bn', correct_slots, keyphrase_mask)
            correct_null_slots = torch.where(correct_null_slots < 0,
                                             correct_null_slots.\
                                             new_zeros(correct_null_slots.shape),
                                             correct_null_slots).detach()


            exact_correct_slots = (masknull_tokensum == target_tokensum).float()
            exact_correct_slots = torch.einsum('bnl,bn->bnl', exact_correct_slots, keyphrase_mask)

            present_correct_slots_num = exact_correct_slots[:, :mid_idx, :mid_idx].sum(-1)
            absent_correct_slots_num = exact_correct_slots[:, mid_idx:, mid_idx:].sum(-1)
            correct_slots_num = torch.cat([present_correct_slots_num,\
                                          absent_correct_slots_num], dim=-1)

            #correct_null_slots = (correct_null_slots + (correct_slots_num == 1).float()).detach()


            exact_correct_slots = torch.nonzero(exact_correct_slots)
            one_matrix = correct_null_slots.new_ones(correct_null_slots.shape)
            # allow cross-label assignment
            for index, exact_correct_slot in enumerate(exact_correct_slots):
                batch_i = exact_correct_slot[0]
                slot_i = exact_correct_slot[1]
                reassign_i = exact_correct_slot[2]
                if (correct_slots_num[batch_i, slot_i] == 1) and \
                    ((slot_i // mid_idx) == (reassign_i // mid_idx)):
                    if (key_correct_slots[batch_i, reassign_i] == 0):
                        one_matrix[batch_i, slot_i] = 0
                        target[batch_i, slot_i] = target[batch_i, reassign_i]
                        trg_mask[batch_i, slot_i] = trg_mask[batch_i, reassign_i]
            one_matrix = one_matrix.detach()
            correct_null_slots = (correct_null_slots + ((correct_slots_num == 1) & (one_matrix ==0)).float()).detach() 

            
            
            # caculation of original loss            
            state = model.decoder.init_state(memory_bank, src_mask)  # refresh the state
            input_tgt = torch.cat([y_t_init, target[:, :, :-1]], dim=-1)
            input_tgt = input_tgt.masked_fill(input_tgt.gt(opt.vocab_size - 1), word2idx[io.UNK_WORD])
            decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov, control_embed)



            forward_time = time_since(start_time)
            forward_time_total += forward_time
            
            
            decoder_dist_reshape = decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)
            decoder_dist_ = decoder_dist_reshape[:, :, 0, :]
            decoder_dist_first_token_probability, decoder_dist_first_token = decoder_dist_.max(-1)
            decoder_dist_null_probability =  decoder_dist_[:, :, word2idx[io.NULL_WORD]]
            decoder_dist_first_token_rate =  decoder_dist_null_probability


            decoder_dist_first_mask = (decoder_dist_first_token == word2idx[io.NULL_WORD])
            target_first_token = target[:, :, 0]
            target_first_token_mask = (target_first_token == word2idx[io.NULL_WORD])
            

            temp = torch.where(target_first_token_mask, decoder_dist_first_token_rate,\
                                decoder_dist_[:, :, word2idx[io.NULL_WORD]]) 


            decoder_dist_ = torch.cat([decoder_dist_[:,:,:word2idx[io.NULL_WORD]],\
                                       temp.unsqueeze(-1), decoder_dist_[:,:,word2idx[io.NULL_WORD]+1:]], dim=-1).unsqueeze(2)


            decoder_dist_reshape = torch.cat([decoder_dist_, decoder_dist_reshape[:, :, 1:, :]], dim=2)
   
            
    
            # scall the loss of token null using the under-estimation of other keyphrase token 
            decoder_dist_predict = decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)
            decoder_dist_predict = torch.gather(decoder_dist_predict, dim=-1, index=target.unsqueeze(-1))
            decoder_dist_predict_ = decoder_dist_predict[:, :, 0, 0]


            decoder_dist_predict_under_estimation = (decoder_dist_predict_ + EPS) / (decoder_dist_null_probability + EPS)
            mask_normal = (decoder_dist_predict_under_estimation >= 1)
            degree_predict_under_estimation = torch.where(mask_normal,\
                                               decoder_dist_predict_under_estimation.\
                                               new_ones(decoder_dist_predict_under_estimation.shape),\
                                               decoder_dist_predict_under_estimation)
            
            decoder_dist_predict_under_estimation = torch.einsum('bn,bn->bn',\
                                            degree_predict_under_estimation,\
                                            1-(target[:,:,0] == word2idx[io.NULL_WORD]).float()).sum(-1) /\
                                            ((1-(target[:,:,0] == word2idx[io.NULL_WORD]).float()).sum(-1) + EPS)
    
            decoder_dist_predict_under_estimation = torch.where(decoder_dist_predict_under_estimation==0,\
                                       decoder_dist_predict_under_estimation.\
                                       new_ones(decoder_dist_predict_under_estimation.shape),\
                                       decoder_dist_predict_under_estimation)

            

    
    
    

            start_time = time.time()
            if opt.fix_kp_num_len:
                if opt.seperate_pre_ab:
                    # loss func of null : p(null)/p(top1_token)
                    mid_idx = opt.max_kp_num // 2
                    pre_loss = masked_cross_entropy(
                        decoder_dist_reshape[:, :mid_idx]\
                            .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                        target[:, :mid_idx].reshape(batch_size, -1),
                        trg_mask[:, :mid_idx].reshape(batch_size, -1),
                        loss_scales=[opt.loss_scale_pre],
                        scale_indices=[word2idx[io.NULL_WORD]],
                        reduce=False)
                    ab_loss = masked_cross_entropy(
                        decoder_dist_reshape[:, mid_idx:]
                            .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                        target[:, mid_idx:].reshape(batch_size, -1),
                        trg_mask[:, mid_idx:].reshape(batch_size, -1),
                        loss_scales=[opt.loss_scale_ab],
                        scale_indices=[word2idx[io.NULL_WORD]],
                        reduce=False)

                    pre_loss = pre_loss.reshape(batch_size, opt.max_kp_num//2, opt.max_kp_len)
                    ab_loss = ab_loss.reshape(batch_size, opt.max_kp_num//2, opt.max_kp_len)

                    loss = torch.cat([pre_loss,ab_loss],dim=1)
                    loss = torch.einsum('bnl,bn->bnl', loss, correct_null_slots)
                    loss_scall = decoder_dist_predict_under_estimation.unsqueeze(-1).unsqueeze(-1).detach() * loss
                    loss = torch.where(target == word2idx[io.NULL_WORD], loss_scall, loss)

                    loss = loss.sum()
                    
                    
                    
                    
                else:
                    loss = masked_cross_entropy(decoder_dist, target.reshape(batch_size, -1),
                                                trg_mask.reshape(batch_size, -1),
                                                loss_scales=[opt.loss_scale], scale_indices=[word2idx[io.NULL_WORD]])
            else:
                loss = masked_cross_entropy(decoder_dist, target, trg_mask)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += trg_mask.sum().item()
            

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def evaluate_greedy_generator(data_loader, generator, opt):
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    interval = 1000
    with torch.no_grad():
        word2idx = opt.vocab['word2idx']
        idx2word = opt.vocab['idx2word']
        start_time = time.time()
        for batch_i, batch in enumerate(data_loader):
            if (batch_i + 1) % interval == 0:
                logging.info("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                start_time = time.time()

            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_idx_list = batch

            if opt.fix_kp_num_len:
                n_best_result = generator.inference(src, src_lens, src_oov, src_mask, oov_lists, word2idx,\
                                                   idx2word)

                pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                     eos_idx=-1,  # to keep all the keyphrases rather than only the first one
                                                     unk_idx=word2idx[io.UNK_WORD],
                                                     replace_unk=opt.replace_unk,
                                                     src_str_list=src_str_list)

                # recover the original order in the dataset
                seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists,
                                       n_best_result['decoder_scores']),
                                   key=lambda p: p[0])
                original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists, decoder_scores = zip(*seq_pairs)

                # Process every src in the batch
                for src_str, trg_str_list, pred, oov, decoder_score in zip(src_str_list, trg_str_2dlist, pred_list,
                                                                           oov_lists, decoder_scores): 
                    all_keyphrase_list = split_word_list_from_set(pred[-1], decoder_score[-1].cpu().numpy(),
                                                                  opt.max_kp_len,
                                                                  opt.max_kp_num, io.EOS_WORD, io.NULL_WORD)
                    # output the predicted keyphrases to a file
                    write_example_kp(pred_output_file, all_keyphrase_list)    
                    
            else:
                n_best_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, word2idx)
                pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                     word2idx[io.EOS_WORD],
                                                     word2idx[io.UNK_WORD],
                                                     opt.replace_unk, src_str_list)

                # recover the original order in the dataset
                seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                                   key=lambda p: p[0])
                original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

                # Process every src in the batch
                for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                    # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                    # pred_seq_list: a list of sequence objects, sorted by scores
                    # oov: a list of oov words
                    # all_keyphrase_list: a list of word list contains all the keyphrases \
                    # in the top max_n sequences decoded by beam search
                    all_keyphrase_list = []
                    for word_list in pred:
                        all_keyphrase_list += split_word_list_by_delimiter(word_list, io.SEP_WORD)

                    # output the predicted keyphrases to a file
                    write_example_kp(pred_output_file, all_keyphrase_list)

    pred_output_file.close()


def write_example_kp(out_file, kp_list):
    pred_print_out = ''
    for word_list_i, word_list in enumerate(kp_list):
        if word_list_i < len(kp_list) - 1:
            pred_print_out += '%s;' % ' '.join(word_list)
        else:
            pred_print_out += '%s' % ' '.join(word_list)
    pred_print_out += '\n'
    out_file.write(pred_print_out)


    

def preprocess_n_best_result(n_best_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                             src_str_list):
    predictions = n_best_result['predictions']
    attention = n_best_result['attention']
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, attn_n_best, oov, src_word_list in zip(predictions, attention, oov_lists, src_str_list):
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            sentences_n_best.append(sentence)
        # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_list.append(sentences_n_best)
    return pred_list
