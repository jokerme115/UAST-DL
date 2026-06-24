import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..config import DEVICE, IMAGE_SIZE, RESULTS_DIR, LB
from ..ust_utils import extract_amp_spectrum, source_to_target_freq
import cv2
from skimage.filters import gaussian
from datetime import datetime
import shutil
import json

plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese labels properly
plt.rcParams['axes.unicode_minus'] = False  # For displaying minus signs properly

class TrainingVisualizer:
    """
    Training process visualization utility class for tracking and displaying image changes during training,
    with special focus on UST-related changes.
    """
    
    def __init__(self, output_dir=None, interval=1, config=None, clear_old=False, use_timestamp=True):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str, optional): Output directory path. Defaults to None.
            interval (int, optional): Visualization interval (save every N epochs). Defaults to 1.
            config (dict, optional): Configuration information, will be saved to the output directory. Defaults to None.
            clear_old (bool, optional): Whether to clear old results. Defaults to False.
            use_timestamp (bool, optional): Whether to append a timestamp to the output directory name. Defaults to True.
        """
        # Set output directory
        if output_dir is None:
            base_dir = os.path.join(RESULTS_DIR, 'visualization')
        else:
            base_dir = output_dir
        
        # Append timestamp
        if use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(base_dir, f'visualization_{timestamp}')
        else:
            self.output_dir = base_dir
        
        # Clear old results
        if clear_old and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Create complete directory structure
        self.create_directory_structure()
        
        # Save configuration information
        if config is not None:
            self.save_config(config)
        
        self.interval = interval
        self.visualization_counter = 0
        self.target_images_cache = None
        
        # Log basic information
        print(f"Visualizer initialized, output directory: {self.output_dir}")
    
    def create_directory_structure(self):
        """
        Create the complete directory structure.
        """
        # Basic directories
        dirs_to_create = [
            self.output_dir,
            os.path.join(self.output_dir, 'original_images'),
            os.path.join(self.output_dir, 'transformed_images'),
            os.path.join(self.output_dir, 'comparison'),
            os.path.join(self.output_dir, 'predictions'),
            os.path.join(self.output_dir, 'ust_changes'),
            # Extended directories
            os.path.join(self.output_dir, 'ust_changes', 'spectrum'),
            os.path.join(self.output_dir, 'ust_changes', 'features'),
            os.path.join(self.output_dir, 'ust_changes', 'pixel_distributions'),
            os.path.join(self.output_dir, 'ust_changes', 'cutmix'),
            os.path.join(self.output_dir, 'metrics'),
            os.path.join(self.output_dir, 'thumbnails')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_config(self, config):
        """
        Save configuration information to a JSON file.
        
        Args:
            config (dict): Configuration information dictionary.
        """
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {config_path}")
    
    def save_summary(self, summary_data):
        """
        Save the training summary report.
        
        Args:
            summary_data (dict): Summary data.
        """
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Training summary saved to: {summary_path}")
    
    def save_thumbnail(self, image, filename, size=(256, 256)):
        """
        Save a thumbnail image.
        
        Args:
            image (np.ndarray): Image data.
            filename (str): Filename.
            size (tuple, optional): Thumbnail size. Defaults to (256, 256).
        """
        # Resize image
        if len(image.shape) == 4:
            image = image[0]  # Take the first sample
        
        if isinstance(image, torch.Tensor):
            image = self._denormalize(image)
        
        # Resize
        thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        
        # Save
        save_path = os.path.join(self.output_dir, 'thumbnails', filename)
        plt.imsave(save_path, thumbnail)
    
    def plot_metrics(self, metrics_dict, filename='training_metrics.png'):
        """
        Plot and save training metrics chart.
        
        Args:
            metrics_dict (dict): Dictionary containing metric names and values.
            filename (str, optional): Filename to save. Defaults to 'training_metrics.png'.
        """
        fig, axes = plt.subplots(len(metrics_dict), 1, figsize=(10, 4 * len(metrics_dict)))
        if len(metrics_dict) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            axes[i].plot(values)
            axes[i].set_title(metric_name)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'metrics', filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Metrics chart saved to: {save_path}")
    
    def _denormalize(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Denormalize an image.
        
        Args:
            tensor (torch.Tensor): Normalized image tensor.
            mean (list, optional): Mean values. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): Standard deviation values. Defaults to [0.229, 0.224, 0.225].
        
        Returns:
            numpy.ndarray: Denormalized image.
        """
        mean = torch.tensor(mean).to(DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(std).to(DEVICE).view(1, 3, 1, 1)
        
        # Denormalize
        denormalized = tensor * std + mean
        
        # Convert to numpy array and clip range to 0-1
        denormalized = denormalized.cpu().detach().numpy().squeeze()
        denormalized = np.clip(denormalized, 0, 1)
        
        # Rearrange dimension order (C, H, W) -> (H, W, C)
        if len(denormalized.shape) == 3:
            denormalized = denormalized.transpose(1, 2, 0)
        
        return denormalized
    
    def visualize_amp_spectrum(self, image, save_path=None):
        """
        Visualize the amplitude spectrum of an image.
        
        Args:
            image (torch.Tensor): Input image.
            save_path (str, optional): Save path. Defaults to None.
            
        Returns:
            numpy.ndarray: Amplitude spectrum image.
        """
        # Convert to numpy array
        img_np = self._denormalize(image)
        if len(img_np.shape) == 2:  # Grayscale image
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        
        # Extract amplitude spectrum
        amp_spectrum = extract_amp_spectrum(img_np.transpose(2, 0, 1))[0]
        
        # Log scale to enhance visualization
        amp_log = np.log(amp_spectrum + 1)
        
        # Normalize
        amp_normalized = []
        for c in range(amp_log.shape[0]):
            amp_c = (amp_log[c] - amp_log[c].min()) / (amp_log[c].max() - amp_log[c].min() + 1e-8)
            amp_normalized.append(amp_c)
        amp_normalized = np.stack(amp_normalized)
        
        # Convert to displayable image
        amp_img = np.fft.fftshift(amp_normalized, axes=(1, 2))
        amp_img = amp_img.transpose(1, 2, 0)
        
        if save_path:
            plt.figure(figsize=(10, 8))
            plt.imshow(amp_img)
            plt.title('Amplitude Spectrum')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return amp_img
    
    def visualize_ust_changes(self, original_image, transformed_image, epoch, batch_idx, mask=None, prediction=None):
        """
        Visualize changes before and after UST transformation.
        
        Args:
            original_image (torch.Tensor): Original image.
            transformed_image (torch.Tensor): Transformed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
            mask (torch.Tensor, optional): Mask. Defaults to None.
            prediction (torch.Tensor, optional): Prediction result. Defaults to None.
        """
        # Only save visualization at the specified interval
        if epoch % self.interval != 0:
            return
        
        # Denormalize images
        original_np = self._denormalize(original_image)
        transformed_np = self._denormalize(transformed_image)
        
        # Compute difference map
        diff = np.abs(original_np - transformed_np)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)  # Normalize
        
        # If mask is provided, denormalize the mask
        mask_np = None
        if mask is not None:
            mask_np = mask.cpu().detach().numpy().squeeze()
        
        # If prediction is provided, convert to labels
        pred_np = None
        if prediction is not None:
            pred_np = torch.argmax(prediction, dim=1).cpu().detach().numpy().squeeze()
        
        # Create subplots
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # Display original image
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display transformed image
        axes[1].imshow(transformed_np)
        axes[1].set_title('Transformed Image')
        axes[1].axis('off')
        
        # Display difference map (highlight UST changes)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('UST Change Heatmap')
        axes[2].axis('off')
        
        # Display mask (if available)
        if mask_np is not None:
            axes[3].imshow(mask_np, cmap='gray')
            axes[3].set_title('Ground Truth Mask')
            axes[3].axis('off')
        else:
            axes[3].axis('off')
        
        # Display prediction (if available)
        if pred_np is not None:
            axes[4].imshow(pred_np, cmap='gray')
            axes[4].set_title('Prediction')
            axes[4].axis('off')
        else:
            axes[4].axis('off')
        
        # Save comparison figure
        save_path = os.path.join(self.output_dir, 'comparison', f'epoch_{epoch}_batch_{batch_idx}.png')
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx} - UST Transformation Effect', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # Save individual original and transformed images
        plt.imsave(os.path.join(self.output_dir, 'original_images', f'epoch_{epoch}_batch_{batch_idx}.png'), original_np)
        plt.imsave(os.path.join(self.output_dir, 'transformed_images', f'epoch_{epoch}_batch_{batch_idx}.png'), transformed_np)
        
        # Save UST change heatmap
        plt.imsave(os.path.join(self.output_dir, 'ust_changes', f'epoch_{epoch}_batch_{batch_idx}.png'), diff, cmap='hot')
        
        # Save prediction (if available)
        if pred_np is not None:
            plt.imsave(os.path.join(self.output_dir, 'predictions', f'epoch_{epoch}_batch_{batch_idx}.png'), pred_np, cmap='gray')
    
    def visualize_cutmix_effect(self, source_image, target_image, cutmix_mask, mixed_image, epoch, batch_idx):
        """
        Visualize the CutMix effect (core part of UCP bidirectional embedding).
        
        Args:
            source_image (torch.Tensor): Source image.
            target_image (torch.Tensor): Target image.
            cutmix_mask (torch.Tensor): CutMix mask.
            mixed_image (torch.Tensor): Mixed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        # Only save visualization at the specified interval
        if epoch % self.interval != 0:
            return
        
        # Denormalize images
        source_np = self._denormalize(source_image)
        target_np = self._denormalize(target_image)
        mixed_np = self._denormalize(mixed_image)
        
        # Convert mask
        mask_np = cutmix_mask.cpu().detach().numpy().squeeze()
        mask_visual = np.stack([mask_np, np.zeros_like(mask_np), np.zeros_like(mask_np)], axis=-1)  # Display mask region in red
        
        # Create subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(source_np)
        axes[0].set_title('Source Image')
        axes[0].axis('off')
        
        axes[1].imshow(target_np)
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        
        axes[2].imshow(mask_visual)
        axes[2].set_title('CutMix Mask')
        axes[2].axis('off')
        
        axes[3].imshow(mixed_np)
        axes[3].set_title('Mixed Image')
        axes[3].axis('off')
        
        # Save CutMix effect visualization
        save_path = os.path.join(self.output_dir, 'ust_changes', f'cutmix_epoch_{epoch}_batch_{batch_idx}.png')
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx} - UCP CutMix Effect', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def create_training_progress_gif(self, epoch):
        """
        Create a GIF animation of training progress.
        
        Args:
            epoch (int): Current epoch.
        """
        import imageio
        
        # Collect comparison images
        comparison_dir = os.path.join(self.output_dir, 'comparison')
        image_paths = [os.path.join(comparison_dir, f) for f in os.listdir(comparison_dir)
                      if f.startswith('epoch_') and f.endswith('.png')]
        image_paths.sort()  # Sort in order
        
        # Limit the number of images to avoid an overly large GIF
        max_images = 30
        if len(image_paths) > max_images:
            image_paths = image_paths[-max_images:]
        
        # Create GIF
        images = []
        for img_path in image_paths:
            img = imageio.imread(img_path)
            images.append(img)
        
        # Save GIF
        gif_path = os.path.join(self.output_dir, f'training_progress_{epoch}.gif')
        imageio.mimsave(gif_path, images, fps=2)
        
        print(f'Training progress GIF saved to: {gif_path}')
    
    def highlight_ust_regions(self, original_image, transformed_image, epoch, batch_idx):
        """
        Highlight UST change regions.
        
        Args:
            original_image (torch.Tensor): Original image.
            transformed_image (torch.Tensor): Transformed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        # Denormalize images
        original_np = self._denormalize(original_image)
        transformed_np = self._denormalize(transformed_image)
        
        # Compute difference and threshold
        diff = np.abs(original_np - transformed_np).mean(axis=2)
        threshold = 0.1  # Difference threshold, adjustable
        highlight_mask = diff > threshold
        
        # Create highlight effect
        highlighted = original_np.copy()
        highlight_color = np.array([1, 0, 0])  # Red highlight
        
        # Apply highlight
        for i in range(highlighted.shape[0]):
            for j in range(highlighted.shape[1]):
                if highlight_mask[i, j]:
                    # Overlay highlight color on the original image
                    highlighted[i, j] = (highlighted[i, j] * 0.5 + highlight_color * 0.5)
        
        # Save highlighted image
        save_path = os.path.join(self.output_dir, 'ust_changes', f'highlight_epoch_{epoch}_batch_{batch_idx}.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(highlighted)
        plt.title(f'Epoch {epoch}, Batch {batch_idx} - UST Change Highlighted Regions')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_spectrum_changes(self, original_image, transformed_image, epoch, batch_idx):
        """
        Visualize spectrum changes (core of TP-RAM).
        
        Args:
            original_image (torch.Tensor): Original image.
            transformed_image (torch.Tensor): Transformed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        # Only save visualization at the specified interval
        if epoch % self.interval != 0:
            return
        
        # Convert to numpy arrays
        original_np = self._denormalize(original_image)
        transformed_np = self._denormalize(transformed_image)
        
        if len(original_np.shape) == 2:
            original_np = np.stack([original_np, original_np, original_np], axis=-1)
        if len(transformed_np.shape) == 2:
            transformed_np = np.stack([transformed_np, transformed_np, transformed_np], axis=-1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Display original and transformed images
        axes[0, 0].imshow(original_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(transformed_np)
        axes[0, 1].set_title('Transformed Image')
        axes[0, 1].axis('off')
        
        # Visualize spectrum for each channel
        for c in range(3):  # RGB channels
            # FFT of original image
            fft_original = np.fft.fft2(original_np[:, :, c])
            fft_shift_original = np.fft.fftshift(fft_original)
            magnitude_spectrum_original = np.log(np.abs(fft_shift_original) + 1)
            
            # FFT of transformed image
            fft_transformed = np.fft.fft2(transformed_np[:, :, c])
            fft_shift_transformed = np.fft.fftshift(fft_transformed)
            magnitude_spectrum_transformed = np.log(np.abs(fft_shift_transformed) + 1)
            
            # Compute spectrum difference
            spectrum_diff = np.abs(magnitude_spectrum_original - magnitude_spectrum_transformed)
            
            # Mark low-frequency region (main area modified by TP-RAM)
            h, w = magnitude_spectrum_original.shape
            center_h, center_w = h // 2, w // 2
            radius = int(min(h, w) * LB)
            
            # Draw a circle on the spectrum plot to indicate the low-frequency region
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            
            # Copy spectrum plots for marking
            spectrum_original_marked = magnitude_spectrum_original.copy()
            spectrum_transformed_marked = magnitude_spectrum_transformed.copy()
            
            # Apply colormap changes to the low-frequency region
            spectrum_original_marked[mask] = (spectrum_original_marked[mask] - spectrum_original_marked.min()) / \
                                            (spectrum_original_marked.max() - spectrum_original_marked.min() + 1e-8) * 0.5 + 0.5
            spectrum_transformed_marked[mask] = (spectrum_transformed_marked[mask] - spectrum_transformed_marked.min()) / \
                                            (spectrum_transformed_marked.max() - spectrum_transformed_marked.min() + 1e-8) * 0.5 + 0.5
            
            # Display spectrum plots
            if c == 0:  # Only show marked spectrum plots for the first channel
                axes[1, 0].imshow(spectrum_original_marked, cmap='viridis')
                axes[1, 0].set_title('Original Image Spectrum (Low-Freq Region Marked)')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(spectrum_transformed_marked, cmap='viridis')
                axes[1, 1].set_title('Transformed Image Spectrum (Low-Freq Region Marked)')
                axes[1, 1].axis('off')
            
            # Spectrum difference (accumulate average for the last channel)
            if c == 0:  # Initialize
                avg_diff = spectrum_diff
            else:
                avg_diff += spectrum_diff
        
        # Display average spectrum difference
        avg_diff = avg_diff / 3
        axes[1, 2].imshow(avg_diff, cmap='hot')
        axes[1, 2].set_title('Spectrum Difference (TP-RAM Impact)')
        axes[1, 2].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(axes[1, 2].images[0], ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Spectrum Difference Intensity')
        
        # Leave the last subplot empty
        axes[1, 3].axis('off')
        
        # Save spectrum changes visualization
        save_path = os.path.join(self.output_dir, 'ust_changes', f'spectrum_changes_epoch_{epoch}_batch_{batch_idx}.png')
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx} - TP-RAM Spectrum Change Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_feature_evolution(self, model, original_image, transformed_image, epoch, batch_idx):
        """
        Visualize the feature evolution process (bidirectional supervision effect of SymGD).
        
        Args:
            model (torch.nn.Module): Model.
            original_image (torch.Tensor): Original image.
            transformed_image (torch.Tensor): Transformed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        # Only save visualization at the specified interval
        if epoch % self.interval != 0:
            return
        
        # Ensure model is in evaluation mode
        was_training = model.training
        model.eval()
        
        # Register hooks to get intermediate features
        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
        
        # Try to get encoder features (assuming segmentation_models_pytorch Unet)
        try:
            # Register hooks for different encoder layers
            encoder_hooks = []
            for i, layer in enumerate(model.model.encoder.encoder.layer):
                hook = layer.register_forward_hook(get_features(f'layer_{i}'))
                encoder_hooks.append(hook)
            
            # Forward pass to get features
            with torch.no_grad():
                original_image = original_image.unsqueeze(0).to(DEVICE) if len(original_image.shape) == 3 else original_image.to(DEVICE)
                transformed_image = transformed_image.unsqueeze(0).to(DEVICE) if len(transformed_image.shape) == 3 else transformed_image.to(DEVICE)
                
                # Get original image features
                model(original_image)
                original_features = {k: v.clone() for k, v in features.items()}
                
                # Get transformed image features
                features.clear()
                model(transformed_image)
                transformed_features = {k: v.clone() for k, v in features.items()}
            
            # Create feature evolution visualization
            num_layers = min(4, len(original_features))  # Show at most 4 layers
            fig, axes = plt.subplots(num_layers, 3, figsize=(15, 5 * num_layers))
            
            for i, layer_name in enumerate(list(original_features.keys())[:num_layers]):
                # Get feature maps (first channel of the first sample)
                orig_feat = original_features[layer_name][0, 0].cpu().numpy()
                trans_feat = transformed_features[layer_name][0, 0].cpu().numpy()
                
                # Normalize feature maps
                orig_feat = (orig_feat - orig_feat.min()) / (orig_feat.max() - orig_feat.min() + 1e-8)
                trans_feat = (trans_feat - trans_feat.min()) / (trans_feat.max() - trans_feat.min() + 1e-8)
                
                # Compute feature difference
                feat_diff = np.abs(orig_feat - trans_feat)
                feat_diff = (feat_diff - feat_diff.min()) / (feat_diff.max() - feat_diff.min() + 1e-8)
                
                # Upsample to the same size for display
                orig_feat = cv2.resize(orig_feat, (256, 256), interpolation=cv2.INTER_LINEAR)
                trans_feat = cv2.resize(trans_feat, (256, 256), interpolation=cv2.INTER_LINEAR)
                feat_diff = cv2.resize(feat_diff, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                # Display feature maps
                if num_layers == 1:
                    ax_row = axes
                else:
                    ax_row = axes[i]
                
                ax_row[0].imshow(orig_feat, cmap='viridis')
                ax_row[0].set_title(f'Original Image Features {layer_name}')
                ax_row[0].axis('off')
                
                ax_row[1].imshow(trans_feat, cmap='viridis')
                ax_row[1].set_title(f'Transformed Image Features {layer_name}')
                ax_row[1].axis('off')
                
                # Use heatmap to highlight differences
                im = ax_row[2].imshow(feat_diff, cmap='hot')
                ax_row[2].set_title(f'Feature Difference {layer_name}')
                ax_row[2].axis('off')
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax_row[2], orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Feature Difference Intensity')
            
            # Save feature evolution visualization
            save_path = os.path.join(self.output_dir, 'ust_changes', f'feature_evolution_epoch_{epoch}_batch_{batch_idx}.png')
            plt.suptitle(f'Epoch {epoch}, Batch {batch_idx} - SymGD Feature Evolution Analysis', fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Feature evolution visualization failed: {e}")
        finally:
            # Remove hooks
            for hook in encoder_hooks:
                hook.remove()
            
            # Restore model state
            if was_training:
                model.train()
    
    def create_ust_change_animation(self, epoch, batch_indices=None):
        """
        Create a UST change animation showing the dynamic effect of changes throughout training.
        
        Args:
            epoch (int): Current epoch.
            batch_indices (list, optional): List of batch indices to include. Defaults to None.
        """
        try:
            import imageio
            
            # Collect UST change heatmaps
            ust_dir = os.path.join(self.output_dir, 'ust_changes')
            image_patterns = [f for f in os.listdir(ust_dir) if f.startswith('epoch_') and '_batch_' in f and not f.endswith('highlight.png') and not f.endswith('cutmix.png') and not f.endswith('spectrum_changes.png') and not f.endswith('feature_evolution.png')]
            
            # If batch_indices is specified, filter images
            if batch_indices is not None:
                filtered_images = []
                for img_name in image_patterns:
                    # Extract batch index
                    batch_str = img_name.split('_batch_')[1].split('.')[0]
                    try:
                        batch_idx = int(batch_str)
                        if batch_idx in batch_indices:
                            filtered_images.append(img_name)
                    except ValueError:
                        continue
                image_patterns = filtered_images
            
            # Sort by epoch and batch index
            def sort_key(img_name):
                epoch_str = img_name.split('epoch_')[1].split('_batch_')[0]
                batch_str = img_name.split('_batch_')[1].split('.')[0]
                return (int(epoch_str), int(batch_str))
            
            image_patterns.sort(key=sort_key)
            
            # Limit the number of images
            max_images = 50
            if len(image_patterns) > max_images:
                image_patterns = image_patterns[-max_images:]
            
            # Create animation
            images = []
            for img_name in image_patterns:
                img_path = os.path.join(ust_dir, img_name)
                img = imageio.imread(img_path)
                
                # Add epoch and batch info to the image
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                epoch_str = img_name.split('epoch_')[1].split('_batch_')[0]
                batch_str = img_name.split('_batch_')[1].split('.')[0]
                text = f'Epoch: {epoch_str}, Batch: {batch_str}'
                cv2.putText(img, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
            
            # Save animation
            animation_path = os.path.join(self.output_dir, f'ust_changes_animation_{epoch}.gif')
            imageio.mimsave(animation_path, images, fps=3)
            
            print(f'UST change animation saved to: {animation_path}')
            
        except Exception as e:
            print(f"Failed to create UST change animation: {e}")
    
    def visualize_pixel_value_distribution(self, original_image, transformed_image, epoch, batch_idx):
        """
        Visualize pixel value distribution changes.
        
        Args:
            original_image (torch.Tensor): Original image.
            transformed_image (torch.Tensor): Transformed image.
            epoch (int): Current epoch.
            batch_idx (int): Current batch index.
        """
        # Only save visualization at the specified interval
        if epoch % self.interval != 0:
            return
        
        # Denormalize images
        original_np = self._denormalize(original_image)
        transformed_np = self._denormalize(transformed_image)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display pixel value distributions for original and transformed images
        colors = ['red', 'green', 'blue']
        labels = ['R', 'G', 'B']
        
        for i in range(3):  # RGB channels
            if len(original_np.shape) == 2:
                # Grayscale image
                axes[0].hist(original_np.ravel(), bins=50, alpha=0.5, color='gray', label='Grayscale')
                axes[1].hist(transformed_np.ravel(), bins=50, alpha=0.5, color='gray', label='Grayscale')
                break
            else:
                # Color image
                axes[0].hist(original_np[:, :, i].ravel(), bins=50, alpha=0.5, color=colors[i], label=labels[i])
                axes[1].hist(transformed_np[:, :, i].ravel(), bins=50, alpha=0.5, color=colors[i], label=labels[i])
        
        axes[0].set_title('Original Image Pixel Value Distribution')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        axes[1].set_title('Transformed Image Pixel Value Distribution')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # Compute and display pixel value difference distribution
        diff = np.abs(original_np - transformed_np)
        if len(diff.shape) == 3:
            diff = diff.mean(axis=2)  # Compute mean difference
        
        axes[2].hist(diff.ravel(), bins=50, alpha=0.7, color='purple')
        axes[2].set_title('Pixel Value Difference Distribution')
        axes[2].set_xlabel('Difference Value')
        axes[2].set_ylabel('Frequency')
        
        # Save distribution visualization
        save_path = os.path.join(self.output_dir, 'ust_changes', f'pixel_distribution_epoch_{epoch}_batch_{batch_idx}.png')
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx} - Pixel Value Distribution Change Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
