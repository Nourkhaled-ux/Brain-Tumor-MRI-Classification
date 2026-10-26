import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('resnet50_best.h5')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

class_info = {
    'glioma'    : 'Glioma — A tumor that starts in the glial cells of the brain.',
    'meningioma': 'Meningioma — A tumor that forms in the meninges (brain membranes).',
    'no tumor'   : 'No Tumor — No brain tumor detected.',
    'pituitary' : 'Pituitary Tumor — A tumor in the pituitary gland.'
}

def predict(image):
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  

    
    preds = model.predict(img_array, verbose=0)[0]    
    results = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    top_idx  = np.argmax(preds)
    top_class = class_names[top_idx]
    confidence = preds[top_idx] * 100
    info = class_info[top_class]

    return results, f"**{top_class.upper()}** ({confidence:.1f}%)\n\n{info}"


with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown("""
    # Brain Tumor MRI Classifier
    Upload a brain MRI scan to classify the tumor type.
    **Classes:** Glioma | Meningioma | No Tumor | Pituitary
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload MRI Scan",
                type="numpy"
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            label_output = gr.Label(
                label="Prediction Probabilities",
                num_top_classes=4
            )
            text_output = gr.Markdown(label="Result")

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, text_output]
    )

    gr.Markdown("### Example Images")
    gr.Examples(
        examples=[],  
        inputs=image_input
    )

demo.launch(share=True) 