import pandas as pd
from datetime import datetime

# Define the file path and name
file_path = "/Users/cazandraaporbo/Desktop/Job_Hunting/job-search-history.xlsx"

# Define the columns for the Excel sheet
columns = [
    "Company Name",
    "Date Applied",
    "Where Applied (e.g., LinkedIn, Company Website)",
    "Job Title",
    "Salary Range",
    "Response Received (Yes/No)",
    "Interviewed (Yes/No)",
    "Interview Date",
    "Interview Outcome (e.g., Passed, Rejected, Pending)",
    "Company Information (e.g., Industry, Size, Location)",
    "Contact Person",
    "Contact Email",
    "Follow-Up Date",
    "Notes",
    "Days Since Applied",  # Calculated column
    "Application Status",  # Calculated column
]

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns)

# Add pre-loaded Excel formulas and calculations
# For example, "Days Since Applied" will calculate the number of days since the application date
# "Application Status" will provide a status based on whether you received a response or were interviewed

# Save the DataFrame to an Excel file
with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Job Search History", index=False)
    
    # Access the XlsxWriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets["Job Search History"]
    
    # Add Excel formulas and formatting
    # Formula for "Days Since Applied" (Column O)
    worksheet.write_formula(1, 14, '=IF(B2<>"", TODAY()-B2, "")')  # Row 2, Column O (index 14)
    
    # Formula for "Application Status" (Column P)
    worksheet.write_formula(1, 15, '=IF(F2="Yes", "Response Received", IF(G2="Yes", "Interviewed", "No Response"))')  # Row 2, Column P (index 15)
    
    # Add conditional formatting for "Application Status"
    status_format = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})  # Red for "No Response"
    worksheet.conditional_format("P2:P1000", {
        "type": "text",
        "criteria": "containing",
        "value": "No Response",
        "format": status_format,
    })
    
    # Add a header format
    header_format = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "top",
        "fg_color": "#4F81BD",
        "font_color": "white",
        "border": 1,
    })
    
    # Write the header row with formatting
    for col_num, value in enumerate(df.columns):
        worksheet.write(0, col_num, value, header_format)
    
    # Set column widths
    worksheet.set_column("A:A", 25)  # Company Name
    worksheet.set_column("B:B", 15)  # Date Applied
    worksheet.set_column("C:C", 30)  # Where Applied
    worksheet.set_column("D:D", 25)  # Job Title
    worksheet.set_column("E:E", 15)  # Salary Range
    worksheet.set_column("F:F", 20)  # Response Received
    worksheet.set_column("G:G", 15)  # Interviewed
    worksheet.set_column("H:H", 15)  # Interview Date
    worksheet.set_column("I:I", 20)  # Interview Outcome
    worksheet.set_column("J:J", 30)  # Company Information
    worksheet.set_column("K:K", 20)  # Contact Person
    worksheet.set_column("L:L", 25)  # Contact Email
    worksheet.set_column("M:M", 15)  # Follow-Up Date
    worksheet.set_column("N:N", 40)  # Notes
    worksheet.set_column("O:O", 15)  # Days Since Applied
    worksheet.set_column("P:P", 20)  # Application Status

print(f"Excel file created successfully at {file_path}")