import streamlit as st
import nibabel as nib
import numpy as np
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tempfile
import os

# Function to load the Keras model (cached for performance)
@st.cache_resource
def load_segmentation_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

MODEL_PATH = "model.h5"
model = load_segmentation_model(MODEL_PATH)

# Function to preprocess the NIfTI image
def preprocess_image(nifti_img, target_size=(128, 128)):
    img_data = nifti_img.get_fdata()
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    middle_slice_idx = img_data.shape[-1] // 2
    slice_2d = img_data[..., middle_slice_idx]

    slice_2d_resized = tf.image.resize(np.expand_dims(slice_2d, axis=-1), target_size)
    processed_input = np.repeat(slice_2d_resized, 4, axis=-1)
    processed_input = np.expand_dims(processed_input, axis=0)

    return processed_input, nifti_img.affine, nifti_img.header, img_data.shape, slice_2d

# Function to post-process model prediction
def postprocess_prediction(prediction, original_affine, original_header, original_shape):
    segmented_slice_2d = prediction[0, ..., 0]

    preview_display_size = (256, 256)
    segmented_slice_for_preview = tf.image.resize(
        np.expand_dims(segmented_slice_2d, axis=-1), preview_display_size
    ).numpy()[..., 0]

    binary_mask_for_preview = (segmented_slice_for_preview > 0.5).astype(np.float32)

    original_slice_h = original_shape[0]
    original_slice_w = original_shape[1]
    resized_prediction_slice = tf.image.resize(
        np.expand_dims(segmented_slice_2d, axis=-1), (original_slice_h, original_slice_w)
    ).numpy()[..., 0]
    binary_mask_3d_slice = (resized_prediction_slice > 0.5).astype(np.int16)

    dummy_segmented_3d_data = np.zeros(original_shape, dtype=np.int16)
    middle_slice_idx = original_shape[-1] // 2
    dummy_segmented_3d_data[..., middle_slice_idx] = binary_mask_3d_slice * 1

    output_nifti = nib.Nifti1Image(dummy_segmented_3d_data, original_affine, original_header)

    return output_nifti, binary_mask_for_preview

# Generate PNG bytes from a 2D image array
def generate_preview_image_bytes(image_array):
    buf = io.BytesIO()
    plt.figure(figsize=(3, 3), dpi=100)
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf.getvalue()

# --- Streamlit UI ---
st.set_page_config(page_title="NIfTI Segmentation App", layout="centered")

st.title("NIfTI Segmentation App")
st.markdown("""
Upload your `.nii` or `.nii.gz` file to get a segmentation prediction.
This application uses your `model.h5` for actual inference and displays a 2D preview of the result.
""")

if model is None:
    st.error("Segmentation model could not be loaded. Please ensure 'model.h5' is in the correct directory.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a .nii or .nii.gz file",
    type=["nii", "nii.gz"],
    help="Select your medical image file for segmentation."
)

# Session states
for key in ['output_nii_data', 'preview_image_bytes', 'input_preview_bytes', 'error_message']:
    if key not in st.session_state:
        st.session_state[key] = None

if st.button("Predict Segmentation", type="primary", disabled=uploaded_file is None):
    st.session_state['error_message'] = ""
    st.session_state['output_nii_data'] = None
    st.session_state['preview_image_bytes'] = None
    st.session_state['input_preview_bytes'] = None

    if uploaded_file is None:
        st.session_state['error_message'] = "Please upload a .nii or .nii.gz file first."
        st.rerun()

    with st.spinner("Processing..."):
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
        tmp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            nifti_img = nib.load(tmp_file_path)
            processed_input, affine, header, original_shape, input_slice = preprocess_image(nifti_img)
            prediction = model.predict(processed_input)
            output_nifti, preview_mask = postprocess_prediction(prediction, affine, header, original_shape)

            # Save output NIfTI to temp file, read bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as output_tmp:
                nib.save(output_nifti, output_tmp.name)
                output_tmp.seek(0)
                st.session_state.output_nii_data = output_tmp.read()
            os.remove(output_tmp.name)

            st.session_state.preview_image_bytes = generate_preview_image_bytes(preview_mask)
            st.session_state.input_preview_bytes = generate_preview_image_bytes(input_slice)

            st.success("Prediction complete!")

        except Exception as e:
            st.session_state['error_message'] = f"An error occurred during prediction: {e}"
            st.error(f"Prediction failed: {e}")
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# Show any error
if st.session_state['error_message']:
    st.error(st.session_state['error_message'])

# Show results
if st.session_state.output_nii_data:
    st.subheader("Segmentation Preview (Input vs Output)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Input Image Slice**")
        st.image(st.session_state.input_preview_bytes, caption="Original Input Slice", width=250)

    with col2:
        st.write("**Predicted Segmentation Mask**")
        st.image(st.session_state.preview_image_bytes, caption="Predicted Output Mask", width=250)

    st.write("### Download Full Segmentation:")
    st.download_button(
        label="Download Segmented Output (.nii)",
        data=st.session_state.output_nii_data,
        file_name="segmented_output.nii",
        mime="application/octet-stream",
        help="Click to download the full 3D segmentation result as a NIfTI file."
    )

st.markdown("---")
st.markdown(
    """
    <p style='font-size: small; color: gray;'>
    This preview shows a 2D slice of the input and the predicted segmentation.
    For full 3D visualization, use specialized medical imaging software.
    </p>
    """,
    unsafe_allow_html=True
)
