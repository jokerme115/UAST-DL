import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_MIN_AREA = 0.1
DEFAULT_MORPH_KERNEL_SIZE = 5
DEFAULT_MORPH_ITERATIONS = 1


def get_deeplab_transform():
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_with_deeplab(model, image, device, transform, use_tta=True):
    with torch.no_grad():
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.fromarray(image)
        orig_w, orig_h = pil_img.size
        new_w = ((orig_w + 15) // 16) * 16
        new_h = ((orig_h + 15) // 16) * 16
        if new_w != orig_w or new_h != orig_h:
            padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            padded.paste(pil_img, (0, 0))
            proc_img = padded
        else:
            proc_img = pil_img
        img_tensor = transform(proc_img).unsqueeze(0).to(device)
        if use_tta:
            pred = model(img_tensor)
            img_tensor_flipped = torch.flip(img_tensor, [-1])
            pred_flipped = model(img_tensor_flipped)
            pred_flipped = torch.flip(pred_flipped, [-1])
            img_tensor_vflipped = torch.flip(img_tensor, [-2])
            pred_vflipped = model(img_tensor_vflipped)
            pred_vflipped = torch.flip(pred_vflipped, [-2])
            pred = (pred + pred_flipped + pred_vflipped) / 3.0
            pred = pred.max(1)[1].cpu().numpy()[0]
        else:
            pred = model(img_tensor).max(1)[1].cpu().numpy()[0]
        pred = pred[:orig_h, :orig_w]
    return pred


def overlay_predictions_on_image(
    original_image, prediction_mask, color=(0, 255, 0), alpha=DEFAULT_OVERLAY_ALPHA
):
    if isinstance(original_image, Image.Image):
        image_array = np.array(original_image)
    else:
        image_array = original_image
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    color_mask = np.zeros_like(image_array)
    color_mask[prediction_mask == 1] = color
    overlay = cv2.addWeighted(image_array, 1 - alpha, color_mask, alpha, 0)
    mask_uint8 = (prediction_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= DEFAULT_MIN_AREA
    ]
    cv2.drawContours(overlay, filtered_contours, -1, color, 2)
    return overlay


def get_cassava_area_mask(
    image,
    min_area=DEFAULT_MIN_AREA,
    morph_kernel_size=DEFAULT_MORPH_KERNEL_SIZE,
    morph_iterations=DEFAULT_MORPH_ITERATIONS,
):
    if isinstance(image, Image.Image):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_bgr = (
            image
            if len(image.shape) == 3 and image.shape[2] == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        )
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_hsv = np.array([10, 30, 30])
    upper_hsv = np.array([40, 255, 255])
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    lower_lab = np.array([120, 90, 120])
    upper_lab = np.array([240, 150, 180])
    mask_lab = cv2.inRange(img_lab, lower_lab, upper_lab)
    lower_ycrcb = np.array([70, 50, 120])
    upper_ycrcb = np.array([230, 140, 190])
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]
    mask_rgb = (
        (
            (r_channel > 40)
            & (r_channel < 255)
            & (g_channel > 40)
            & (g_channel < 255)
            & (b_channel > 20)
            & (b_channel < 230)
        ).astype(np.uint8)
        * 255
    )
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)
    combined_mask = cv2.bitwise_or(combined_mask, mask_rgb)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, small_kernel, iterations=morph_iterations
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=morph_iterations)
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_mask = np.zeros_like(combined_mask)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_area_threshold = img_bgr.shape[0] * img_bgr.shape[1] * 0.005
        for i, contour in enumerate(contours[:10]):
            area = cv2.contourArea(contour)
            if area > min_area_threshold and area >= min_area:
                cv2.fillPoly(final_mask, [contour], 255)
            elif i == 0 and area >= min_area:
                cv2.fillPoly(final_mask, [contour], 255)
    if np.count_nonzero(final_mask) < (img_bgr.shape[0] * img_bgr.shape[1] * 0.01):
        mask_relaxed = (
            (
                (r_channel > 20)
                & (r_channel < 255)
                & (g_channel > 20)
                & (g_channel < 255)
                & (b_channel > 10)
                & (b_channel < 240)
            ).astype(np.uint8)
            * 255
        )
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_relaxed = cv2.morphologyEx(
            mask_relaxed, cv2.MORPH_CLOSE, kernel_large, iterations=morph_iterations
        )
        mask_relaxed = cv2.morphologyEx(
            mask_relaxed, cv2.MORPH_OPEN, small_kernel, iterations=morph_iterations
        )
        contours, _ = cv2.findContours(
            mask_relaxed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= min_area:
                cv2.fillPoly(final_mask, [largest_contour], 255)
    final_mask = cv2.morphologyEx(
        final_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )
    result = (final_mask > 0).astype(np.uint8)
    return result
