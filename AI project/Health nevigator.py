import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.tree import DecisionTreeClassifier


# Load and process the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path).dropna()
        symptoms = data['Symptoms'].str.get_dummies(sep=', ')  # One-hot encode the symptoms
        X = symptoms
        y = data['Disease']
        medications = data[['Disease', 'Medicine']].drop_duplicates().set_index('Disease').to_dict()['Medicine']
        return X, y, medications, symptoms.columns.tolist()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return None, None, None, None


# Train the model
def train_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


# Convert selected symptoms into a binary vector matching the training data columns
def get_input_vector(selected_symptoms, all_symptoms):
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    return pd.DataFrame([input_vector], columns=all_symptoms)  # Return DataFrame to ensure feature names alignment


# Diagnosis Function
def diagnose():
    selected_symptoms = [symptom for symptom, var in symptom_vars.items() if var.get()]

    if not selected_symptoms:
        messagebox.showwarning("Input Error", "Please select at least one symptom.")
        return

    # Convert selected symptoms into a binary DataFrame matching the order of training columns
    input_data = get_input_vector(selected_symptoms, all_symptoms)

    # Predict the disease
    predicted_disease = model.predict(input_data)[0]

    # Retrieve the suggested medication for the predicted disease
    medication = medications.get(predicted_disease, "No medication found.")

    # Display the result
    result_text = f"Predicted Disease: {predicted_disease}\n\nSuggested Medication: {medication}\n\nAdvice: Please consult a doctor for confirmation."
    result_label.config(text=result_text)

    # Show the buttons for new prediction and exit after diagnosis
    new_prediction_button.pack(side="left", padx=10, pady=10)
    exit_button.pack(side="left", padx=10, pady=10)


# New prediction function
def new_prediction():
    # Reset symptom selection
    for var in symptom_vars.values():
        var.set(False)

    result_label.config(text="")  # Clear the previous results
    new_prediction_button.pack_forget()  # Hide the "New Prediction" button until after diagnosis
    exit_button.pack_forget()  # Hide the "Exit" button until after diagnosis


# Exit application function
def exit_application():
    root.destroy()


# Create a simple symptom list with a search bar
def create_symptom_box(root):
    symptom_frame = ttk.Frame(root)
    symptom_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

    # Search Bar
    search_label = ttk.Label(symptom_frame, text="Search Symptoms:")
    search_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
    search_entry = ttk.Entry(symptom_frame, textvariable=search_var, width=30)
    search_entry.grid(row=0, column=1, padx=10, pady=5)
    search_entry.bind("<KeyRelease>", lambda e: update_symptoms())

    # Scrollable Frame for Symptoms
    canvas = tk.Canvas(symptom_frame)
    canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")
    scrollbar = ttk.Scrollbar(symptom_frame, orient="vertical", command=canvas.yview)
    scrollbar.grid(row=1, column=2, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    symptom_list_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=symptom_list_frame, anchor="nw")

    # Dynamically create checkbuttons for symptoms
    for idx, symptom in enumerate(all_symptoms):
        var = tk.BooleanVar()
        checkbutton = ttk.Checkbutton(symptom_list_frame, text=symptom, variable=var, width=25)
        checkbutton.grid(row=idx, column=0, padx=10, pady=5, sticky="w")
        symptom_vars[symptom] = var

    # Configure canvas scroll region
    symptom_list_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


# Update the symptoms list based on the search query
def update_symptoms():
    query = search_var.get().lower()
    for widget in symptom_widgets.values():
        widget.grid_forget()  # Hide all symptoms initially

    for idx, symptom in enumerate(all_symptoms):
        if query in symptom.lower():  # If query matches symptom
            symptom_widgets[symptom].grid(row=idx, column=0, padx=10, pady=5, sticky="w")  # Show matched symptoms


# Main GUI Setup
def setup_gui():
    global root  # Make root accessible in other functions
    root = tk.Tk()
    root.title("Health Navigator")
    root.geometry("700x700")  # Spacious UI
    root.resizable(False, False)

    # Initialize search_var after the root is created
    global search_var
    search_var = tk.StringVar()

    # Title
    title_label = ttk.Label(root, text="Health Navigator", font=("Arial", 18, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Instructions
    instruction_label = ttk.Label(root, text="Select your symptoms from the list below:")
    instruction_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Symptom Box
    create_symptom_box(root)

    # Predict Button
    predict_button = ttk.Button(root, text="Predict Disease", command=diagnose)
    predict_button.grid(row=4, column=0, columnspan=2, pady=20, sticky="ew")

    # Result Area
    global result_label
    result_label = ttk.Label(root, text="", font=("Arial", 12), wraplength=500, justify="left")
    result_label.grid(row=5, column=0, columnspan=2, pady=10)

    # Frame for Buttons (New Prediction and Exit)
    button_frame = ttk.Frame(root)
    button_frame.grid(row=6, column=0, columnspan=2, pady=10)

    # New Prediction Button
    global new_prediction_button
    new_prediction_button = ttk.Button(button_frame, text="New Prediction", command=new_prediction)
    new_prediction_button.pack_forget()  # Initially hidden

    # Exit Button
    global exit_button
    exit_button = ttk.Button(button_frame, text="Exit", command=exit_application)
    exit_button.pack_forget()  # Initially hidden

    root.mainloop()


# Load data and initialize
file_path = r"C:\Users\AHMAR HASSAN\Desktop\medical data.csv"  # Update the path to your dataset
X, y, medications, all_symptoms = load_data(file_path)
if X is not None and y is not None:
    model = train_model(X, y)
    symptom_vars = {}
    symptom_widgets = {}
    setup_gui()
else:
    print("Error: Dataset could not be loaded.")
