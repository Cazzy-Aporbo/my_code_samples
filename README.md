# **Code Samples Portfolio**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-D4E4FC?style=for-the-badge&logo=python&logoColor=7393B3)](https://python.org)
[![Data Viz](https://img.shields.io/badge/Data%20Visualization-FFE5E5?style=for-the-badge&logo=plotly&logoColor=CD919E)](https://seaborn.pydata.org)
[![Excel Automation](https://img.shields.io/badge/Excel%20Automation-E7F3E7?style=for-the-badge&logo=microsoftexcel&logoColor=5C8A5C)](https://openpyxl.readthedocs.io)
[![GUI Development](https://img.shields.io/badge/GUI%20Development-F3E5F5?style=for-the-badge&logo=python&logoColor=9C89B8)](https://docs.python.org/3/library/tkinter.html)
[![Data Analysis](https://img.shields.io/badge/Data%20Analysis-FFF4E6?style=for-the-badge&logo=pandas&logoColor=FFA500)](https://pandas.pydata.org)

</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-D4E4FC?style=flat-square&color=D4E4FC)
![separator](https://img.shields.io/badge/-FFE5E5?style=flat-square&color=FFE5E5)
![separator](https://img.shields.io/badge/-E7F3E7?style=flat-square&color=E7F3E7)
![separator](https://img.shields.io/badge/-F3E5F5?style=flat-square&color=F3E5F5)
![separator](https://img.shields.io/badge/-FFF4E6?style=flat-square&color=FFF4E6)

</div>

### **Professional Python Solutions for Data Management and Visualization**

This repository showcases three carefully crafted Python applications that demonstrate practical solutions for everyday data challenges. Each project emphasizes clean code architecture, user-friendly interfaces, and professional-grade output. I've designed these tools to be both functional and educational, serving as templates for similar applications while maintaining production-ready quality.

<div align="center">
  
![separator](https://img.shields.io/badge/-FFEFD5?style=flat&color=FFEFD5)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0FFF0?style=flat&color=F0FFF0)

</div>

---

## **Project Portfolio**

<table>
<tr style="background-color:#D4E4FC;">
<td><strong>Project</strong></td>
<td><strong>Category</strong></td>
<td><strong>Key Technologies</strong></td>
<td><strong>Primary Purpose</strong></td>
</tr>
<tr style="background-color:#FFE5E5;">
<td><strong>Job Application Tracker</strong></td>
<td>Productivity Tool</td>
<td>Pandas, OpenPyXL, XlsxWriter</td>
<td>Automated Excel generation with dynamic calculations and conditional formatting</td>
</tr>
<tr style="background-color:#E7F3E7;">
<td><strong>Advanced Seaborn Visualizations</strong></td>
<td>Data Visualization</td>
<td>Seaborn, Matplotlib, NumPy</td>
<td>Sophisticated statistical visualizations with publication-quality aesthetics</td>
</tr>
<tr style="background-color:#F3E5F5;">
<td><strong>Productivity Tracker</strong></td>
<td>Personal Analytics</td>
<td>Tkinter, Seaborn, Pandas</td>
<td>GUI-based productivity logging with heatmap visualization</td>
</tr>
</table>

---

## **Job Application Tracker**

### **Project Overview**

I developed this Excel automation tool to streamline the job search process, eliminating manual spreadsheet setup and providing intelligent tracking features. The application generates a professionally formatted Excel workbook with automated calculations and visual indicators for application status.

<table>
<tr style="background-color:#B3D9FF;">
<td><strong>Feature</strong></td>
<td><strong>Implementation Details</strong></td>
</tr>
<tr style="background-color:#CCE5FF;">
<td><strong>Automated File Generation</strong></td>
<td>Creates structured Excel workbook with predefined columns and formatting</td>
</tr>
<tr style="background-color:#E0F0FF;">
<td><strong>Dynamic Calculations</strong></td>
<td>Auto-calculates days since application and updates status based on responses</td>
</tr>
<tr style="background-color:#F0F8FF;">
<td><strong>Conditional Formatting</strong></td>
<td>Highlights pending applications in red, completed in green for visual tracking</td>
</tr>
<tr style="background-color:#E6F3FF;">
<td><strong>Professional Layout</strong></td>
<td>Optimized column widths and formatted headers for readability</td>
</tr>
</table>

### **Technical Specifications**

<table>
<tr style="background-color:#FFE8E8;">
<td><strong>Component</strong></td>
<td><strong>Description</strong></td>
</tr>
<tr style="background-color:#FFF0F0;">
<td><strong>Core Libraries</strong></td>
<td><code>pandas</code> for data manipulation, <code>openpyxl</code> for Excel operations, <code>xlsxwriter</code> for advanced formatting</td>
</tr>
<tr style="background-color:#FFF5F5;">
<td><strong>Data Structure</strong></td>
<td>Company name, job title, salary range, response status, interview details, follow-up notes</td>
</tr>
<tr style="background-color:#FFFAFA;">
<td><strong>Output Format</strong></td>
<td>Excel file with multiple sheets, formulas, and visual formatting</td>
</tr>
<tr style="background-color:#FFF8F8;">
<td><strong>File Location</strong></td>
<td><code>/Users/cazandraaporbo/Desktop/Job_Hunting/job-search-history.xlsx</code></td>
</tr>
</table>

### **Installation & Usage**

```bash
# Install dependencies
pip install pandas openpyxl xlsxwriter

# Run the application
python make_file_copy.py
```

---

## **Advanced Seaborn Visualizations**

### **Project Overview**

This visualization suite demonstrates advanced statistical graphics capabilities, generating publication-quality visualizations from dynamically created datasets. I designed this to showcase sophisticated data exploration techniques while maintaining aesthetic appeal through carefully selected color palettes and layout designs.

<table>
<tr style="background-color:#E8F5E8;">
<td><strong>Visualization Type</strong></td>
<td><strong>Purpose</strong></td>
<td><strong>Color Palette</strong></td>
</tr>
<tr style="background-color:#F0F9F0;">
<td><strong>Violin Plot</strong></td>
<td>Distribution analysis across categories with density estimation</td>
<td>Coolwarm gradient</td>
</tr>
<tr style="background-color:#F5FDF5;">
<td><strong>Pair Plot</strong></td>
<td>Multi-dimensional correlation matrix with regression lines</td>
<td>Default Seaborn</td>
</tr>
<tr style="background-color:#F8FFF8;">
<td><strong>Hexbin Heatmap</strong></td>
<td>Density visualization for large datasets with binning</td>
<td>Inferno</td>
</tr>
<tr style="background-color:#FAFFFA;">
<td><strong>2D KDE Plot</strong></td>
<td>Continuous density estimation for bivariate analysis</td>
<td>Viridis</td>
</tr>
</table>

### **Dataset Characteristics**

<table>
<tr style="background-color:#F0E6FF;">
<td><strong>Parameter</strong></td>
<td><strong>Configuration</strong></td>
</tr>
<tr style="background-color:#F5F0FF;">
<td><strong>Sample Size</strong></td>
<td>1000 dynamically generated data points</td>
</tr>
<tr style="background-color:#F8F5FF;">
<td><strong>Categories</strong></td>
<td>Technology, Art, Music, Sports, Science, Fashion</td>
</tr>
<tr style="background-color:#FAF8FF;">
<td><strong>Subcategories</strong></td>
<td>Multiple nested categories (AI, Jazz, Physics, etc.)</td>
</tr>
<tr style="background-color:#FCFAFF;">
<td><strong>Metrics</strong></td>
<td>Popularity (normal), Engagement (uniform), Uniqueness (integer)</td>
</tr>
</table>

### **Key Features**

- Dynamic dataset generation with realistic distributions
- Multiple visualization techniques in a single workflow
- Professional color schemes optimized for presentation
- Annotated axes and titles for clarity
- Grid styling for enhanced readability

### **Installation & Usage**

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn

# Run the visualization suite
python advanced_seaborn_viz.py
```

---

## **Productivity Tracker**

### **Project Overview**

I created this GUI-based application to provide an intuitive way to track daily productivity levels with visual feedback. The system combines a user-friendly Tkinter interface with powerful data visualization capabilities, storing data persistently for long-term trend analysis.

<table>
<tr style="background-color:#FFF4E6;">
<td><strong>Component</strong></td>
<td><strong>Functionality</strong></td>
</tr>
<tr style="background-color:#FFF8F0;">
<td><strong>GUI Interface</strong></td>
<td>Tkinter-based form with date selection and productivity scale (1-10)</td>
</tr>
<tr style="background-color:#FFFAF5;">
<td><strong>Data Persistence</strong></td>
<td>CSV storage for historical tracking and analysis</td>
</tr>
<tr style="background-color:#FFFCFA;">
<td><strong>Visualization Engine</strong></td>
<td>Seaborn heatmap showing productivity patterns over time</td>
</tr>
<tr style="background-color:#FFFEFD;">
<td><strong>Interactive Controls</strong></td>
<td>Log button for data entry, Show Heatmap for instant visualization</td>
</tr>
</table>

### **Technical Architecture**

<table>
<tr style="background-color:#E6E6FA;">
<td><strong>Layer</strong></td>
<td><strong>Technology</strong></td>
<td><strong>Purpose</strong></td>
</tr>
<tr style="background-color:#F0F0FF;">
<td><strong>Presentation</strong></td>
<td>Tkinter</td>
<td>User interface with input validation</td>
</tr>
<tr style="background-color:#F5F5FF;">
<td><strong>Business Logic</strong></td>
<td>Python Core</td>
<td>Date processing and productivity calculations</td>
</tr>
<tr style="background-color:#F8F8FF;">
<td><strong>Data Layer</strong></td>
<td>Pandas + CSV</td>
<td>Persistent storage and data manipulation</td>
</tr>
<tr style="background-color:#FAFAFF;">
<td><strong>Visualization</strong></td>
<td>Matplotlib + Seaborn</td>
<td>Heatmap generation with color-coded productivity levels</td>
</tr>
</table>

### **User Workflow**

1. **Data Entry**: Input date in YYYY-MM-DD format and select productivity level
2. **Logging**: Click "Log Productivity" to save entry to CSV
3. **Visualization**: Click "Show Heatmap" to generate visual representation
4. **Analysis**: Review patterns to identify high and low productivity periods

### **Installation & Usage**

```bash
# Install dependencies
pip install pandas matplotlib seaborn

# Note: Tkinter typically comes pre-installed with Python
# Run the application
python productivity_tracker.py
```

---

## **Development Environment**

<table>
<tr style="background-color:#D4F1D4;">
<td><strong>Requirement</strong></td>
<td><strong>Specification</strong></td>
</tr>
<tr style="background-color:#E0F5E0;">
<td><strong>Python Version</strong></td>
<td>3.8 or higher recommended</td>
</tr>
<tr style="background-color:#E8F8E8;">
<td><strong>Operating System</strong></td>
<td>Cross-platform (Windows, macOS, Linux)</td>
</tr>
<tr style="background-color:#F0FAF0;">
<td><strong>IDE Recommendation</strong></td>
<td>VS Code, PyCharm, or Jupyter for visualization projects</td>
</tr>
<tr style="background-color:#F5FCF5;">
<td><strong>Memory Requirements</strong></td>
<td>Minimal (< 100MB for all applications)</td>
</tr>
</table>

---

## **Common Customization Options**

Each project has been designed with extensibility in mind. Here are suggested modifications:

<table>
<tr style="background-color:#FFE0CC;">
<td><strong>Project</strong></td>
<td><strong>Customization Possibilities</strong></td>
</tr>
<tr style="background-color:#FFE8DD;">
<td><strong>Job Tracker</strong></td>
<td>Add interview scheduling, company research notes, salary negotiation tracking</td>
</tr>
<tr style="background-color:#FFF0E8;">
<td><strong>Visualizations</strong></td>
<td>Implement real-time data feeds, add interactive Plotly graphs, create dashboard views</td>
</tr>
<tr style="background-color:#FFF5F0;">
<td><strong>Productivity</strong></td>
<td>Include mood tracking, integrate with calendar APIs, add goal-setting features</td>
</tr>
</table>

---

## **Troubleshooting Guide**

<table>
<tr style="background-color:#D8BFD8;">
<td><strong>Issue</strong></td>
<td><strong>Solution</strong></td>
</tr>
<tr style="background-color:#E6D6E6;">
<td>ImportError for libraries</td>
<td>Ensure all dependencies are installed via pip</td>
</tr>
<tr style="background-color:#F0E8F0;">
<td>File path errors</td>
<td>Update paths to match your system structure</td>
</tr>
<tr style="background-color:#F5F0F5;">
<td>GUI not displaying</td>
<td>Verify Tkinter installation and display settings</td>
</tr>
<tr style="background-color:#FAF5FA;">
<td>Visualization rendering issues</td>
<td>Check matplotlib backend configuration</td>
</tr>
</table>

---

## **Future Enhancements**

I'm continuously improving these applications. Planned updates include:

- **Cloud Integration**: AWS/Azure storage for data persistence
- **Web Interfaces**: Flask/Django implementations for browser access
- **Machine Learning**: Predictive analytics for job search and productivity patterns
- **Mobile Compatibility**: React Native versions for mobile tracking

---

## **License & Usage**

<div align="center">

These projects are open-source and available for modification and distribution. Feel free to adapt them for your specific needs or use them as learning resources.

</div>

---

<div align="center">

<strong>Crafted with attention to detail for practical data solutions</strong>

![separator](https://img.shields.io/badge/-D4E4FC?style=flat-square&color=D4E4FC)
![separator](https://img.shields.io/badge/-FFE5E5?style=flat-square&color=FFE5E5)
![separator](https://img.shields.io/badge/-E7F3E7?style=flat-square&color=E7F3E7)
![separator](https://img.shields.io/badge/-F3E5F5?style=flat-square&color=F3E5F5)
![separator](https://img.shields.io/badge/-FFF4E6?style=flat-square&color=FFF4E6)

</div>
