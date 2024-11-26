import tkinter as tk
from tkinter import scrolledtext
import subprocess

venv_activate = "..\\..\\Scripts\\activate && "

def run_script(script_name):
    try:
        result = subprocess.run(f"{venv_activate} python {script_name}", shell=True, capture_output=True, text=True, check=True)
        output_text.delete(1.0, tk.END) 
        output_text.insert(tk.END, result.stdout)
    except subprocess.CalledProcessError as e:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Error: {e.stderr}")

#window
root = tk.Tk()
root.title("[Demo] Disease Prediction Menu")
root.geometry("600x600")

# Labels and Entry widgets for input fields on the left side
tk.Label(root, text="Patient Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
patient_name_entry = tk.Entry(root, width=40)
patient_name_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Patient Age:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
patient_age_entry = tk.Entry(root, width=40)
patient_age_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Other Medical Fields:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
patient_id_entry = tk.Entry(root, width=40)
patient_id_entry.grid(row=2, column=1, padx=10, pady=5)

# Buttons for each script on the right side
btn_kidney = tk.Button(root, text="Check for Kidney Disease", width=20, command=lambda: run_script("kidney.py"))
btn_kidney.grid(row=0, column=2, padx=20, pady=10)

btn_diabetes = tk.Button(root, text="Check for Diabetes", width=20, command=lambda: run_script("diabetes.py"))
btn_diabetes.grid(row=1, column=2, padx=20, pady=10)

btn_heart = tk.Button(root, text="Check for Heart Disease", width=20, command=lambda: run_script("heart_pso.py"))
btn_heart.grid(row=2, column=2, padx=20, pady=10)

output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, font=("Arial", 10))
output_text.grid(row=3, column=0, columnspan=3, padx=10, pady=20)

root.mainloop()
