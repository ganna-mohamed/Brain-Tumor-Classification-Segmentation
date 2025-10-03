import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# إعداد الصفحة
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Brain MRI Tumor Segmentation")
st.write(" Upload Your MRI Here")

# -----------------------------
# الدوال المساعدة
# -----------------------------
# function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

# function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

# function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@st.cache_resource
def load_segmentation_model():
    model = load_model(
        "Brain_Tumor_Seg.h5",
        custom_objects={
            "iou_coef": iou_coef, "dice_coef": dice_coef , "dice_loss" : dice_loss}
    )
    return model

model = load_segmentation_model()

# -----------------------------
# معالجة الصورة
# -----------------------------
def preprocess_image(img, target_size=( 256, 256)):
    img = cv2.resize(img, target_size)
    if len(img.shape) == 2:  # لو رمادي نحوله لـ 3 قنوات
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1,H,W,3)
    return img

def predict_mask(img, threshold=0.3):
    pred = model.predict(img, verbose=0)
    pred_mask = (pred[0,:,:,0] > threshold).astype(np.uint8)
    return pred_mask

# -----------------------------
# واجهة المستخدم
# -----------------------------
# -------------------- Improved UI for segmentation preview --------------------
import io
import base64

uploaded_file = st.file_uploader("📤 Upload MRI", type=["jpg", "jpeg", "png", "tif", "tiff"])

# Sidebar controls for segmentation visualization
st.sidebar.header("Visualization settings")
threshold = st.sidebar.slider("⚙ Threshold to extract the tumor", 0.0, 1.0, 0.3, 0.01)
overlay_opacity = st.sidebar.slider("🔆 Overlay opacity", 0.0, 1.0, 0.6, 0.01)
display_mode = st.sidebar.selectbox("🖼 Display mode", ["Mask only", "Overlay", "Heatmap", "Contour"])
overlay_color = st.sidebar.selectbox("🎨 Overlay color", ["Red", "Green", "Blue", "Yellow"])
download_png = st.sidebar.checkbox("Enable download buttons", value=True)

# helper: color map selection -> BGR tuple for cv2
COLOR_MAPS = {
    "Red": (0,0,255),
    "Green": (0,255,0),
    "Blue": (255,0,0),
    "Yellow": (0,255,255)
}

def pil_to_bytes(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def make_download_link(img_pil, filename="result.png"):
    b = pil_to_bytes(img_pil)
    b64 = base64.b64encode(b).decode()
    href = f"data:file/png;base64,{b64}"
    return href

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("📌 Original MRI")
    st.image(image_array, caption="Original MRI", use_column_width=250)

    # run prediction with spinner
    with st.spinner("Running segmentation..."):
        try:
            X = preprocess_image(image_array)          # افتراض وجود الدالة في الكود الأساسي
            pred_mask = predict_mask(X, threshold)     # افتراض وجود الدالة في الكود الأساسي
        except Exception as e:
            st.error("❌ Error during prediction:")
            st.exception(e)
            pred_mask = None

    if pred_mask is not None:
        # Ensure mask is 2D binary numpy array (values 0/1)
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
            pred_mask = np.squeeze(pred_mask, axis=-1)
        # Sometimes model outputs (H,W) normalized floats; convert to binary
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # resize mask to original size
        mask_resized = cv2.resize(pred_mask.astype(np.uint8),
                                  (image_array.shape[1], image_array.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # compute area metrics
        tumor_pixels = int(mask_resized.sum())
        total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
        tumor_percent = tumor_pixels / total_pixels * 100

        # create colored overlay
        color = COLOR_MAPS.get(overlay_color, (0,0,255))  # BGR for opencv
        overlay = image_array.copy()
        # apply color only where mask==1
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[mask_resized == 1] = color[::-1]  # convert BGR->RGB by reversing tuple
        # blend using opacity
        blended = (overlay * (1 - overlay_opacity) + overlay_mask * overlay_opacity).astype(np.uint8)

        # prepare heatmap if requested
        heatmap_img = None
        if display_mode == "Heatmap":
            # create colored heatmap from probabilities if pred_mask had float probs; else use mask
            try:
                # if pred_mask originally had probabilities (not just binary), try to use scaled float
                probs = cv2.resize(pred_mask.astype(np.float32),
                                   (image_array.shape[1], image_array.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
                hm = (probs * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                heatmap_img = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            except Exception:
                heatmap_img = blended

        # prepare contour image
        contour_img = None
        if display_mode == "Contour":
            # find contours on mask (cv2 expects uint8)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = image_array.copy()
            cv2.drawContours(contour_img, contours, -1, color[::-1], thickness=2)

        # UI layout: two columns for results + controls below
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("🩻 Tumor Mask")
            st.caption(f"Pixels: {tumor_pixels} — {tumor_percent:.4f}% of image")
            mask_vis = (mask_resized * 255).astype(np.uint8)
            st.image(mask_vis, use_column_width=250, clamp=True)

            if download_png:
                pil_mask = Image.fromarray(mask_vis)
                href = make_download_link(pil_mask, filename="tumor_mask.png")
                st.markdown(f"[⬇ Download mask]({href})")

        with col2:
            st.subheader("🖼 Result")
            if display_mode == "Mask only":
                st.image(mask_vis, caption="Mask only", use_column_width=250)
            elif display_mode == "Overlay":
                st.image(blended, caption=f"Overlay (opacity={overlay_opacity})", use_column_width=True)
                if download_png:
                    pil_overlay = Image.fromarray(blended)
                    href2 = make_download_link(pil_overlay, filename="overlay.png")
                    st.markdown(f"[⬇ Download overlay]({href2})")
            elif display_mode == "Heatmap":
                st.image(heatmap_img, caption="Heatmap", use_column_width=True)
                if download_png:
                    pil_hm = Image.fromarray(heatmap_img)
                    href3 = make_download_link(pil_hm, filename="heatmap.png")
                    st.markdown(f"[⬇ Download heatmap]({href3})")
            elif display_mode == "Contour":
                st.image(contour_img, caption="Contour overlay", use_column_width=True)
                if download_png:
                    pil_ct = Image.fromarray(contour_img)
                    href4 = make_download_link(pil_ct, filename="contour.png")
                    st.markdown(f"[⬇ Download contour image]({href4})")

        # show small statistics panel
        st.markdown("---")
        st.write("### Summary")
        st.write(f"- Tumor pixels: *{tumor_pixels}*")
        st.write(f"- Image size: *{mask_resized.shape[1]} x {mask_resized.shape[0]}* (W x H)")
        st.write(f"- Tumor area: *{tumor_percent:.4f}%*")

    else:
        st.warning("No mask produced.")
    # -----------------------------
    # Classification
    # -----------------------------
    tumor_pixels = np.sum(pred_mask_resized)
    if tumor_pixels > 50:  # لو فيه بكسلات كفاية
        classification = "🟥 Tumor Detected"
    else:
        classification = "🟩 No Tumor"

    st.subheader("📌 Classification Result")
    st.write(classification)
    st.write("Tumor Pixels:", tumor_pixels)

    # -----------------------------
    # عرض النتائج
    # -----------------------------
    st.subheader("Result: ")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_array, caption="Original MRI")

    with col2:
        st.image(pred_mask_resized*255, caption="Predicted Mask", clamp=True)

    with col3:
        st.image(overlay, caption="Overlay")

else:
    st.info("⬆ Upload MRI To continue")