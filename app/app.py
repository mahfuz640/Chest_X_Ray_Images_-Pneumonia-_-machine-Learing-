import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk

# --- মডেল লোড ---
model = tf.keras.models.load_model("pneumonia_model.h5")

# --- ইমেজ প্রসেস ফাংশন ---
def preprocess_image(img):
    img = img.convert("L")               # গ্রেস্কেল
    img = img.resize((224, 224))         # আপনার ট্রেনিং সাইজ
    img_array = np.array(img) / 255.0    # normalize
    img_array = np.expand_dims(img_array, axis=-1)   # (224,224,1)
    img_array = np.expand_dims(img_array, axis=0)    # (1,224,224,1)
    return img_array

# --- ফাইল সিলেক্ট ---
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img_tk = ImageTk.PhotoImage(img.resize((250, 250)))

    panel.config(image=img_tk)
    panel.image = img_tk
    panel.file_path = file_path

# --- প্রেডিকশন ---
def predict():
    if not hasattr(panel, "file_path"):
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    img = Image.open(panel.file_path)
    img_array = preprocess_image(img)

    prediction = model.predict(img_array)
    result = np.argmax(prediction, axis=1)[0]

    if result == 0:   # Normal
        messagebox.showinfo("Result", "✅ আলহামদুলিল্লাহ আপনি সুস্থ আছেন\n✅ Alhamdulliah apni sustho asen")
    else:             # Pneumonia
        messagebox.showerror("Result", "⚠️ ডাক্তার বা বিশেষজ্ঞের পরামর্শ নিন\n⚠️ Doctor ba bishesh ogger poramorsho nin")

# --- Tkinter UI ---
root = tk.Tk()
root.title("নিউমোনিয়া শনাক্তকরণ সিস্টেম (Tkinter)")
root.geometry("400x450")

btn_upload = tk.Button(root, text="📂 X-ray আপলোড করুন", command=upload_image)
btn_upload.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

btn_predict = tk.Button(root, text="🔍 Result", command=predict, bg="green", fg="white", font=("Arial", 12, "bold"))
btn_predict.pack(pady=10)

footer = tk.Label(root, text="👨‍💻 Creator: Engr. Md. Mahfuzur Rahman", fg="blue")
footer.pack(side="bottom", pady=5)

root.mainloop()
