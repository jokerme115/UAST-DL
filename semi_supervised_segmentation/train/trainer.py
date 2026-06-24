import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import copy
import segmentation_models_pytorch as smp
from semi_supervised_segmentation import config
from semi_supervised_segmentation.evaluate.metrics import evaluate_model_detailed, evaluate_model_sliding_full
from semi_supervised_segmentation.utils.utils_semi import update_ema_variables, get_current_consistency_weight, obtain_cutmix_box, TPRAM_Mixer
from semi_supervised_segmentation.utils.logger import append_epoch_metrics
from semi_supervised_segmentation.train.losses_semi import CE_Dice_FocalTverskyLoss, FocalTverskyLoss, DiceLossWithMask, FocalLoss

def train_model(model, train_loader, val_loader, unlabeled_loader, num_epochs, 
                enable_early_stopping=False, early_stopping_patience=10, early_stopping_monitor='val_miou'):
    """
    Train the model using UST-RUN semi-supervised strategy.
    """
    
    # Setup device
    device = config.DEVICE
    model = model.to(device)
    
    teacher_mode = getattr(config, 'TEACHER_MODE', 'ema')
    if teacher_mode not in {'ema', 'student', 'none'}:
        teacher_mode = 'ema'

    ema_model = None
    if teacher_mode == 'ema':
        ema_model = copy.deepcopy(model)
        ema_model = ema_model.to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.detach_()
    
    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler
    total_iters = num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.PolyLR(optimizer, max_iters=total_iters, power=0.9) if hasattr(optim.lr_scheduler, 'PolyLR') else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss Functions Initialization
    sup_ce_loss = str(getattr(config, 'SUP_CE_LOSS', 'ce')).lower()
    if sup_ce_loss in {'focal', 'focalloss'}:
        focal_alpha = getattr(config, 'FOCAL_ALPHA', 0.75)
        focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)
        criterion_ce = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=255).to(device)
    else:
        sup_ce_loss = 'ce'
        criterion_ce = nn.CrossEntropyLoss(ignore_index=255).to(device)
    
    # Tversky / Focal Tversky
    tversky_alpha = getattr(config, 'TVERSKY_ALPHA', 0.3)
    tversky_beta = getattr(config, 'TVERSKY_BETA', 0.7)
    # Default to gamma=1.0 for standard Tversky if not specified
    tversky_gamma = getattr(config, 'TVERSKY_GAMMA', 1.0) 
    criterion_tversky = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma).to(device)
    
    # Dice
    dice_weights = getattr(config, 'DICE_CLASS_WEIGHTS', [1.0, 10.0])
    criterion_dice = DiceLossWithMask(n_classes=config.NUM_CLASSES, weight=dice_weights).to(device)
    
    # Loss Weights
    w_ce = getattr(config, 'CE_WEIGHT', 0.5)
    w_tversky = getattr(config, 'TVERSKY_WEIGHT', 1.0)
    w_dice = getattr(config, 'DICE_WEIGHT', 0.0) # Default 0 if using Tversky
    
    criterion_unsup = nn.MSELoss().to(device)
    
    # TP-RAM Mixer
    tpram_enabled = bool(getattr(config, 'TPRAM_ENABLED', False))
    tpram_mixer = TPRAM_Mixer(L=0.01, device=device) if tpram_enabled else None
    
    # Amp Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    best_miou = 0.0
    best_iou_c1 = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    train_history = {'loss': [], 'miou': [], 'precision': []}
    val_history = {'loss': [], 'miou': [], 'precision': []}
    
    print(f"Start training for {num_epochs} epochs with UST-RUN strategy (Refined)...")
    print(f"Loss Config: CE({sup_ce_loss})={w_ce}, Tversky={w_tversky} (a={tversky_alpha},b={tversky_beta}), Dice={w_dice}")
    print(f"Teacher Mode: {teacher_mode}")
    
    global_step = 0
    
    # Unsupervised Params
    unsupervised_weight = getattr(config, 'UNSUPERVISED_WEIGHT', None)
    if unsupervised_weight is None:
        unsupervised_weight = getattr(config, 'CONSISTENCY_WEIGHT', 10.0)
    rampup_epochs = getattr(config, 'UNSUP_RAMPUP_EPOCHS', None)
    if rampup_epochs is None:
        rampup_epochs = getattr(config, 'CONSISTENCY_RAMPUP', 5)
    ema_decay = getattr(config, 'EMA_DECAY', 0.999)
    warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
    has_explicit_confidence_threshold = hasattr(config, 'CONFIDENCE_THRESHOLD')
    confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', None)
    if confidence_threshold is None:
        confidence_threshold = getattr(config, 'THRESHOLD', 0.0)
    threshold_c0 = getattr(config, 'THRESHOLD_C0', None)
    threshold_c1 = getattr(config, 'THRESHOLD_C1', None)
    ust_mask_mode = getattr(config, 'UST_MASK_MODE', 'class_balance')
    bg_weight = float(getattr(config, 'UST_BG_WEIGHT', 0.1))
    min_pseudo_conf = float(getattr(config, 'MIN_PSEUDO_CONF', 0.0))
    pseudo_label_strategy = str(getattr(config, 'PSEUDO_LABEL_STRATEGY', 'argmax')).lower()
    pseudo_fg_topk = float(getattr(config, 'PSEUDO_FG_TOPK', 0.0))
    unsup_loss_type = str(getattr(config, 'UNSUP_LOSS_TYPE', getattr(config, 'UNSUPERVISED_LOSS_TYPE', 'mse'))).lower()
    supervised_weight = float(getattr(config, 'SUPERVISED_WEIGHT', 1.0))

    max_train_batches = getattr(config, 'DEBUG_MAX_TRAIN_BATCHES', None)
    skip_validation = bool(getattr(config, 'DEBUG_SKIP_VALIDATION', False))
    
    iter_unlabeled = iter(unlabeled_loader) if unlabeled_loader else None
    
    for epoch in range(num_epochs):
        model.train()
        if ema_model is not None:
            ema_model.eval()

        lr_epoch = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        
        train_loss = 0.0
        train_loss_sup = 0.0
        train_loss_unsup = 0.0
        
        # Calculate current consistency weight
        if epoch < warmup_epochs:
            current_unsup_weight = 0.0
        else:
            current_unsup_weight = get_current_consistency_weight(epoch - warmup_epochs, float(unsupervised_weight), float(rampup_epochs))
        
        print(f"Epoch {epoch+1}: Unsupervised Weight: {current_unsup_weight:.4f}")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs_l, targets_l) in enumerate(pbar):
            inputs_l = inputs_l.to(device)
            targets_l = targets_l.to(device).long().squeeze(1)
            
            # Fetch Unlabeled Data
            inputs_u_w, inputs_u_s = None, None
            if iter_unlabeled:
                try:
                    u_data = next(iter_unlabeled)
                except StopIteration:
                    iter_unlabeled = iter(unlabeled_loader)
                    u_data = next(iter_unlabeled)
                
                if isinstance(u_data, (list, tuple)) and len(u_data) == 2:
                    inputs_u_w, inputs_u_s = u_data
                    inputs_u_w = inputs_u_w.to(device)
                    inputs_u_s = inputs_u_s.to(device)
                else:
                    inputs_u_w = u_data.to(device)
                    inputs_u_s = u_data.to(device)
            
            optimizer.zero_grad()
            mask_u = None
            
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                # 1. Supervised Loss
                outputs_l = model(inputs_l)
                
                loss_sup_ce = criterion_ce(outputs_l, targets_l)
                loss_sup_tversky = criterion_tversky(outputs_l, targets_l)
                loss_sup_dice = criterion_dice(outputs_l, targets_l)
                
                loss_sup = w_ce * loss_sup_ce + w_tversky * loss_sup_tversky + w_dice * loss_sup_dice
                
                loss_unsup = torch.tensor(0.0).to(device)
                
                # 2. Unsupervised Loss
                if inputs_u_w is not None and current_unsup_weight > 0 and teacher_mode != 'none':
                    with torch.no_grad():
                        teacher_logits = None
                        if teacher_mode == 'ema':
                            if ema_model is None:
                                raise RuntimeError('Teacher mode is ema but ema_model is None')
                            teacher_logits = ema_model(inputs_u_w)
                        elif teacher_mode == 'student':
                            was_training = model.training
                            model.eval()
                            teacher_logits = model(inputs_u_w)
                            if was_training:
                                model.train()

                        probs_u_w_teacher = None
                        if teacher_logits is not None:
                            probs_u_w_teacher = torch.softmax(teacher_logits, dim=1)
                    
                    if probs_u_w_teacher is not None:
                        pseudo_label_argmax = probs_u_w_teacher.argmax(dim=1)
                        pseudo_conf = probs_u_w_teacher.gather(1, pseudo_label_argmax.unsqueeze(1)).squeeze(1)
                        pseudo_label = pseudo_label_argmax

                        if config.NUM_CLASSES == 2 and pseudo_label_strategy == 'fg_topk' and pseudo_fg_topk > 0:
                            fg_prob = probs_u_w_teacher[:, 1]
                            b, h, w = fg_prob.shape
                            pseudo_label = torch.full_like(pseudo_label_argmax, 255)
                            mask_topk = torch.zeros((b, h, w), dtype=torch.bool, device=device)
                            k = int(max(1, round(pseudo_fg_topk * h * w)))
                            for bi in range(b):
                                flat = fg_prob[bi].reshape(-1)
                                topk_vals, _ = torch.topk(flat, k=k, largest=True)
                                thr = topk_vals[-1]
                                m = fg_prob[bi] >= thr
                                if min_pseudo_conf > 0:
                                    m = m & (fg_prob[bi] >= min_pseudo_conf)
                                mask_topk[bi] = m
                            pseudo_label = torch.where(mask_topk, torch.ones_like(pseudo_label), pseudo_label)

                        if (threshold_c0 is not None and threshold_c1 is not None and config.NUM_CLASSES == 2 and not has_explicit_confidence_threshold):
                            thr_map = torch.where(
                                pseudo_label_argmax == 0,
                                torch.as_tensor(float(threshold_c0), device=device),
                                torch.as_tensor(float(threshold_c1), device=device),
                            )
                            mask_conf = pseudo_conf >= thr_map
                        elif confidence_threshold and float(confidence_threshold) > 0:
                            mask_conf = pseudo_conf >= float(confidence_threshold)
                        elif min_pseudo_conf and float(min_pseudo_conf) > 0:
                            mask_conf = pseudo_conf >= float(min_pseudo_conf)
                        else:
                            mask_conf = torch.ones_like(pseudo_conf, dtype=torch.bool)

                        if ust_mask_mode == 'foreground':
                            mask_conf = mask_conf & (pseudo_label_argmax == 1)
                            mask_u = mask_conf.float().unsqueeze(1)
                        elif ust_mask_mode == 'max_prob':
                            mask_u = mask_conf.float().unsqueeze(1)
                        else:
                            weights = torch.where(pseudo_label_argmax == 0, torch.as_tensor(bg_weight, device=device), torch.as_tensor(1.0, device=device))
                            mask_u = (mask_conf.float() * weights).unsqueeze(1)

                        if config.NUM_CLASSES == 2 and pseudo_label_strategy == 'fg_topk' and pseudo_fg_topk > 0:
                            mask_u = (pseudo_label != 255).float().unsqueeze(1)

                    # TP-RAM Mixer
                    if tpram_mixer is not None and np.random.random() < 0.5:
                        inputs_u_s_mixed = tpram_mixer(inputs_u_s, inputs_l, epoch, num_epochs)
                    else:
                        inputs_u_s_mixed = inputs_u_s
                        
                    outputs_u_s = model(inputs_u_s_mixed)
                    probs_u_s = torch.softmax(outputs_u_s, dim=1)
                    
                    if probs_u_w_teacher is not None:
                        if unsup_loss_type == 'kl':
                            per_pixel = F.kl_div(torch.log(probs_u_s + 1e-8), probs_u_w_teacher.detach(), reduction='none').sum(dim=1)
                        elif unsup_loss_type == 'ce':
                            per_pixel = F.cross_entropy(outputs_u_s, pseudo_label, reduction='none', ignore_index=255)
                        elif unsup_loss_type == 'dice':
                            loss_unsup = criterion_dice(outputs_u_s, pseudo_label, mask=mask_u)
                            per_pixel = None
                        else:
                            per_pixel = ((probs_u_s - probs_u_w_teacher.detach()) ** 2).mean(dim=1)

                        if per_pixel is not None:
                            if mask_u is not None:
                                weighted = per_pixel * mask_u.squeeze(1)
                                denom_mask = mask_u.sum()
                                if denom_mask.item() > 0:
                                    mask_is_binary = torch.logical_or(
                                        torch.isclose(mask_u, torch.zeros((), device=device)),
                                        torch.isclose(mask_u, torch.ones((), device=device)),
                                    ).all()
                                    if bool(mask_is_binary):
                                        loss_unsup = weighted.mean()
                                    else:
                                        loss_unsup = weighted.sum() / (denom_mask + 1e-6)
                                else:
                                    loss_unsup = torch.tensor(0.0, device=device)
                            else:
                                loss_unsup = per_pixel.mean()

                    loss = supervised_weight * loss_sup + current_unsup_weight * loss_unsup
                else:
                    loss = supervised_weight * loss_sup
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if teacher_mode == 'ema' and ema_model is not None and ema_decay > 0:
                copy_bn_stats = bool(getattr(config, 'EMA_COPY_BN_STATS', False))
                update_ema_variables(model, ema_model, ema_decay, global_step, copy_bn_stats=copy_bn_stats)
            global_step += 1
            
            train_loss += loss.item()
            train_loss_sup += loss_sup.item()
            train_loss_unsup += loss_unsup.item()
            
            pbar.set_postfix({'L': loss.item(), 'Sup': loss_sup.item(), 'Unsup': loss_unsup.item()})

            if epoch == 0 and batch_idx == 0:
                unsup_enabled = bool(inputs_u_w is not None and current_unsup_weight > 0 and teacher_mode != 'none')
                print(f"Debug Unsupervised Enabled: {unsup_enabled}")
                print(f"Debug Threshold: {confidence_threshold}")
                print(f"Debug Mask Mode: {ust_mask_mode}")
                print(f"Debug Unsup Loss: {unsup_loss_type}")
                if mask_u is not None:
                    print(f"Debug Mask Sum: {float(mask_u.sum().item()):.2f}")

            if max_train_batches is not None and (batch_idx + 1) >= int(max_train_batches):
                break
            
        scheduler.step()
        
        denom = len(train_loader)
        if max_train_batches is not None:
            denom = min(denom, int(max_train_batches))
        avg_loss = train_loss / max(denom, 1)
        
        val_metrics_student = {'mIoU': 0.0, 'Loss': 0.0, 'IoU_per_class': [0.0, 0.0]}
        val_miou_student = 0.0
        val_iou_c1_student = 0.0

        val_metrics = None
        val_miou = 0.0
        val_loss = 0.0
        val_iou_c1 = 0.0

        if not skip_validation:
            val_metrics_student = validate_model(model, val_loader, criterion_ce, device)
            val_miou_student = val_metrics_student.get('mIoU', 0.0)
            val_iou_c1_student = val_metrics_student.get('IoU_per_class', [0.0, 0.0])[1] if len(val_metrics_student.get('IoU_per_class', [])) > 1 else 0.0

            if teacher_mode == 'ema' and ema_model is not None:
                val_metrics = validate_model(ema_model, val_loader, criterion_ce, device)
                val_miou = val_metrics.get('mIoU', 0.0)
                val_loss = val_metrics.get('Loss', 0.0)
                if 'IoU_per_class' in val_metrics and len(val_metrics['IoU_per_class']) > 1:
                    val_iou_c1 = val_metrics['IoU_per_class'][1]
            else:
                val_miou = val_miou_student
                val_loss = val_metrics_student.get('Loss', 0.0)
                val_iou_c1 = val_iou_c1_student
        
        train_history['loss'].append(avg_loss)
        train_history['miou'].append(val_miou_student) # Log Student mIoU for history
        val_history['loss'].append(val_loss)
        val_history['miou'].append(val_miou)
        
        denom2 = len(train_loader)
        if max_train_batches is not None:
            denom2 = min(denom2, int(max_train_batches))
        denom2 = max(denom2, 1)
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f} (Sup: {train_loss_sup/denom2:.4f}, Unsup: {train_loss_unsup/denom2:.4f})")
        print(f"Val Student: mIoU: {val_miou_student:.4f}, Fg IoU: {val_iou_c1_student:.4f}")
        if teacher_mode == 'ema' and ema_model is not None:
            print(f"Val EMA:     mIoU: {val_miou:.4f}, Fg IoU: {val_iou_c1:.4f}")

        val_precision_student = float(val_metrics_student.get('Precision', 0.0)) if isinstance(val_metrics_student, dict) else 0.0
        val_precision = float(val_metrics.get('Precision', 0.0)) if isinstance(val_metrics, dict) else val_precision_student
        append_epoch_metrics({
            'Epoch': epoch + 1,
            'LR': lr_epoch,
            'Train_Loss': avg_loss,
            'Train_Loss_Sup': float(train_loss_sup / denom2),
            'Train_Loss_Unsup': float(train_loss_unsup / denom2),
            'Unsup_Weight': float(current_unsup_weight),
            'Teacher_Mode': teacher_mode,
            'Val_Student_mIoU': float(val_miou_student),
            'Val_Student_FgIoU': float(val_iou_c1_student),
            'Val_Student_Precision': val_precision_student,
            'Val_mIoU': float(val_miou),
            'Val_FgIoU': float(val_iou_c1),
            'Val_Loss': float(val_loss),
            'Val_Precision': val_precision,
        })
        
        # Save Best Model (Based on EMA usually, but we can track Student too)
        save = False
        model_to_save = 'student'
        monitor_miou = val_miou_student
        monitor_iou_c1 = val_iou_c1_student
        monitor_loss = val_metrics_student.get('Loss', 0.0)
        if teacher_mode == 'ema' and ema_model is not None:
            ema_monitor_miou = val_miou
            ema_monitor_iou_c1 = val_iou_c1
            ema_monitor_loss = val_loss
            if early_stopping_monitor == 'val_iou_c1':
                if ema_monitor_iou_c1 > monitor_iou_c1:
                    model_to_save = 'ema'
                    monitor_iou_c1 = ema_monitor_iou_c1
            elif early_stopping_monitor == 'val_miou':
                if ema_monitor_miou > monitor_miou:
                    model_to_save = 'ema'
                    monitor_miou = ema_monitor_miou
            else:
                if ema_monitor_loss < monitor_loss:
                    model_to_save = 'ema'
                    monitor_loss = ema_monitor_loss

        if early_stopping_monitor == 'val_miou':
            if monitor_miou > best_miou:
                best_miou = monitor_miou
                save = True
        elif early_stopping_monitor == 'val_iou_c1':
            if monitor_iou_c1 > best_iou_c1:
                best_iou_c1 = monitor_iou_c1
                save = True
        else:
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                save = True
                
        if save:
            save_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            if model_to_save == 'ema' and teacher_mode == 'ema' and ema_model is not None:
                torch.save(ema_model.state_dict(), save_path)
                print(f"Saved best EMA model to {save_path}")
            else:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best Student model to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if enable_early_stopping and patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
                
    return train_history, val_history

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    
    use_sliding = getattr(config, 'USE_SLIDING_WINDOW_EVAL', False)
    
    if use_sliding:
        print("Validating with Sliding Window...")
        window_size = getattr(config, 'SLIDING_WINDOW_SIZE', config.IMAGE_SIZE)
        stride_default = (window_size[0] // 2, window_size[1] // 2)
        stride = getattr(config, 'SLIDING_WINDOW_STRIDE', stride_default)
        
        # Check if dataset has image_paths (Standard Dataset)
        if hasattr(val_loader.dataset, 'image_paths'):
             metrics = evaluate_model_sliding_full(
                model=model,
                image_paths=val_loader.dataset.image_paths,
                label_paths=val_loader.dataset.label_paths,
                device=device,
                num_classes=config.NUM_CLASSES,
                patch_size=window_size,
                stride=stride
            )
        else:
            # Fallback if using a different dataset type or subset
             metrics = evaluate_model_detailed(model, val_loader, criterion, device, num_classes=config.NUM_CLASSES)
    else:
        metrics = evaluate_model_detailed(model, val_loader, criterion, device, num_classes=config.NUM_CLASSES)
        
    return metrics
