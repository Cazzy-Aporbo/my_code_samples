# my_code_samples

# Job Application Tracker

## Overview
The Job Application Tracker is a Python script that helps users manage and track job applications in an organized manner. The program creates an Excel file with structured columns, automates calculations, and applies conditional formatting to enhance usability. It provides an easy way to monitor job applications, responses, and interview outcomes.

## Features
- **Automated Excel File Creation**: Generates an Excel file (`job-search-history.xlsx`) to store job applications.
- **Structured Data Entry**: Includes predefined columns such as company name, job title, salary range, response status, interview details, and follow-up notes.
- **Automated Calculations**:
  - Days since the application was submitted.
  - Application status based on responses and interviews.
- **Conditional Formatting**:
  - Highlights "No Response" applications in red for easy tracking.
  - Formats headers for a clean, professional look.
- **Column Width Adjustment**: Ensures readability by setting optimal column widths.

## Requirements
To run this program, install the required Python libraries:
```
pip install pandas openpyxl xlsxwriter
```

## How It Works
1. **Run the script**: The program creates an Excel file at `/Users/cazandraaporbo/Desktop/Job_Hunting/job-search-history.xlsx`.
2. **Enter job applications manually into the file**.
3. **Automatic Calculations**:
   - `Days Since Applied` updates based on the current date.
   - `Application Status` updates dynamically based on the response and interview status.
4. **View and manage job applications**:
   - Open the Excel file to track progress.
   - Use the formatted fields to monitor follow-ups and interview statuses.

## File Structure
- **`make_file_copy.py`**: Python script that generates the Excel file.
- **`job-search-history.xlsx`**: The generated spreadsheet containing job applications.

## Example Usage
1. **Creating a Job Application Log**:
   - Run the script to generate the Excel file.
   - Open the file and enter job application details.
2. **Tracking Application Status**:
   - The `Days Since Applied` column automatically updates based on the application date.
   - The `Application Status` column dynamically adjusts based on responses and interviews.

## Customization
- Modify `file_path` to save the file in a different directory.
- Add more columns if needed (e.g., recruiter details, networking contacts).
- Adjust the conditional formatting to highlight priority applications.

## Troubleshooting
- **File Not Found**: Ensure the script is executed in an environment with write permissions.
- **Excel File Not Updating**: Re-run the script if you want to generate a fresh template.
- **Formatting Issues**: Open the Excel file and verify column structures.

## License
This project is open-source and can be modified to fit individual job tracking needs.


---

# Advanced Seaborn Visualizations

## Overview
This Python program generates **advanced and colorful visualizations** using Seaborn and Matplotlib. The dataset is dynamically generated with **randomized categorical and numerical values**, allowing for visually rich **data exploration**. The program includes:
- **Violin Plots** to show the distribution of popularity scores across categories.
- **Pair Plots** to observe correlations between multiple variables.
- **Hexbin Heatmaps** to highlight dense clusters in engagement vs. uniqueness.
- **2D KDE Plots** to reveal density variations in engagement vs. popularity.

This script is **perfect for showcasing advanced data visualization techniques**, making it useful for **data scientists, analysts, and educators** who want to explore graphical transformations in Python.

## Features
- **Dynamic Dataset Generation**: Randomized data with multiple categorical and numerical attributes.
- **Multiple Visualization Types**:
  - **Violin Plot** for popularity distribution.
  - **Pair Plot** to visualize relationships between multiple metrics.
  - **Hexbin Heatmap** to highlight engagement vs. uniqueness clusters.
  - **2D KDE Density Plot** for engagement vs. popularity insights.
- **Colorful & Aesthetic Visuals**:
  - Uses `coolwarm`, `inferno`, and `viridis` palettes for stunning effects.
  - Grid styles and annotations enhance readability.

## Installation
Ensure you have Python installed, then install the required dependencies:
```sh
pip install numpy pandas matplotlib seaborn
```

## How It Works
1. **Dataset Generation**:
   - The script generates 1000 data points across six categories: Technology, Art, Music, Sports, Science, and Fashion.
   - Each category has multiple subcategories (e.g., "AI", "Jazz", "Physics").
   - Three numeric values (`Popularity`, `Engagement`, `Uniqueness`) are assigned using normal, uniform, and integer distributions.

2. **Visualization Steps**:
   - **Violin Plot**: Displays the distribution of popularity scores across categories and subcategories.
   - **Pair Plot**: A multi-variable scatterplot matrix to explore relationships between all numeric variables.
   - **Hexbin Heatmap**: Uses a hexagonal binning approach to visualize density between engagement and uniqueness.
   - **2D KDE Density Plot**: Shows a density contour of engagement vs. popularity for deeper insights.

## Example Output
- **Violin Plot**: Highlights distribution differences in popularity between different categories.
- **Pair Plot**: Helps understand how variables relate to one another.
- **Hexbin Heatmap**: Shows dense clusters of engagement vs. uniqueness.
- **KDE Plot**: Displays variations in engagement and popularity.

## Customization Options
- Modify `n=1000` to increase or decrease the number of generated data points.
- Change the **color palette** in Seaborn functions (e.g., use `magma`, `plasma` for alternative aesthetics).
- Adjust the `figsize` parameters for larger or smaller plots.
- Add new categories and subcategories in the `subcategories` dictionary to explore more data types.

## Troubleshooting
- **Plots Not Displaying?** Ensure you are running the script in an interactive Python environment (Jupyter Notebook or a script with `plt.show()`).
- **Missing Data?** If filtering or modifications were made, verify that numerical fields are not empty.
- **Hexbin Plot Appears Blank?** Ensure that `Engagement` and `Uniqueness` values are well-distributed.

## License
This project is open-source and available for modification and distribution. Feel free to customize and improve upon it!

---


# Productivity Tracker

## Overview
The Productivity Tracker is a Python-based application that allows users to log their daily productivity levels and visualize trends using a heatmap. It features a user-friendly graphical interface (GUI) built with Tkinter and uses Matplotlib and Seaborn to generate insightful visualizations. The logged data is stored in a CSV file for persistent tracking.

## Features
- **User Input Interface**: Enter date and productivity level using a simple Tkinter interface.
- **Data Logging**: Stores productivity data in a CSV file for tracking progress.
- **Color-Coded Heatmap**: Generates a visual representation of productivity trends over a given period.
- **Interactive Buttons**: Log productivity and display heatmap directly from the GUI.
- **Persistent Data Storage**: Saves data in `/Users/cazandraaporbo/Desktop/mygit/code_samples/productivity_log.csv` for future reference.

## Requirements
To run this program, install the required Python libraries:
```
pip install pandas matplotlib seaborn tkinter
```

## How It Works
1. **Run the script**: Launch the application by executing the Python file.
2. **Enter productivity data**:
   - Input a date in `YYYY-MM-DD` format.
   - Select a productivity level between 1 and 10.
   - Click **"Log Productivity"** to save the data.
3. **View Productivity Heatmap**:
   - Click **"Show Heatmap"** to visualize trends.
   - The heatmap color-codes productivity levels to show high and low productivity days.

## File Structure
- **`productivity_tracker.py`**: Main script containing the program logic.
- **`productivity_log.csv`**: CSV file where logged data is stored.

## Example Usage
1. **Logging Productivity**:
   - Date: `2025-02-28`
   - Productivity Level: `8`
   - Click **"Log Productivity"**
2. **Generating Heatmap**:
   - Click **"Show Heatmap"** to display trends.

## Customization
- Modify `file_path` in the script to store logs in a different location.
- Adjust the heatmap color scheme by changing the `cmap` parameter in the `sns.heatmap()` function.
- Expand functionality to track additional metrics like mood, exercise, or focus hours.

## Troubleshooting
- **Error: File Not Found**: Ensure `productivity_log.csv` exists in the specified directory.
- **Heatmap Display Issues**: Verify that the dataset contains enough data points for visualization.
- **GUI Not Opening**: Ensure Tkinter is installed and your Python version supports it.

## License
This project is open-source and available for modification and distribution.

