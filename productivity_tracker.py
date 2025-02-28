import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# Define file path for saving logs
file_path = "/Users/cazandraaporbo/Desktop/mygit/code_samples/productivity_log.csv"

# Check if the file exists, if not, create it
if not os.path.exists(file_path):
    df = pd.DataFrame(columns=["Date", "Productivity Level"])
    df.to_csv(file_path, index=False)

# Function to log productivity
def log_productivity():
    date = date_entry.get()
    level = level_var.get()
    
    if not date or not level:
        messagebox.showwarning("Input Error", "Please enter a date and select a productivity level.")
        return
    
    # Append to CSV
    df = pd.read_csv(file_path)
    df = df.append({"Date": date, "Productivity Level": int(level)}, ignore_index=True)
    df.to_csv(file_path, index=False)
    messagebox.showinfo("Success", "Productivity logged successfully!")
    plot_heatmap()

# Function to generate a heatmap
def plot_heatmap():
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    
    pivot_table = df.pivot("Month", "Day", "Productivity Level")
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=True, linewidths=0.5, cbar=True)
    plt.title("Daily Productivity Heatmap")
    plt.xlabel("Day")
    plt.ylabel("Month")
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Productivity Tracker")
root.geometry("400x300")

# Labels & Inputs
tk.Label(root, text="Enter Date (YYYY-MM-DD):").pack()
date_entry = tk.Entry(root)
date_entry.pack()

tk.Label(root, text="Select Productivity Level (1-10):").pack()
level_var = tk.IntVar()
tk.Scale(root, from_=1, to=10, orient="horizontal", variable=level_var).pack()

tk.Button(root, text="Log Productivity", command=log_productivity).pack()
tk.Button(root, text="Show Heatmap", command=plot_heatmap).pack()

root.mainloop()
