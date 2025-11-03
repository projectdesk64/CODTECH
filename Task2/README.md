# Automated Data Profiling Report Generator

An intelligent Python tool that automatically generates professional PDF reports from your data files. It performs comprehensive data analysis, creates visualizations, and optionally includes AI-powered executive summaries.

## Features

- **Multi-Format Support**: Works with CSV, Excel (.xlsx, .xls), and JSON files
- **Automatic Data Detection**: Automatically finds and processes the latest data file in the directory
- **Comprehensive Analysis**: 
  - Dataset overview and statistics
  - Detailed numeric column analysis with descriptive statistics
  - Categorical column frequency analysis
  - Visual analysis with histograms and bar charts
  - Data samples for quick inspection
- **AI-Powered Summaries**: Optional executive summaries generated using OpenRouter AI (requires API key)
- **Professional PDF Output**: Clean, formatted PDF reports with headers, footers, and styled tables
- **Automatic File Organization**: All reports are saved in a dedicated `reports/` folder
- **Auto-Open Reports**: Generated PDFs are automatically opened in your default PDF viewer

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The script requires the following packages:
- `pandas` - Data manipulation and analysis
- `reportlab` - PDF generation
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualizations
- `openpyxl` - Excel file support
- `requests` - API requests for AI summaries
- `urllib3` - HTTP client utilities
- `python-dotenv` - Environment variable management

## Configuration

To enable AI-powered executive summaries, you'll need an OpenRouter API key:

1. Create a `.env` file in the project root directory
2. Add your OpenRouter API key:
```
OPENROUTER_API_KEY="your_api_key_here"
```

**Note**: You can get a free API key from [OpenRouter](https://openrouter.ai/). The AI summary feature uses the free Mistral 7B model by default.


```bash
python report_generator.py
```

### With AI Summary

To generate a report with an AI-powered executive summary:

```bash
python report_generator.py --use-ai
```

### Specify Input File

To process a specific data file:

```bash
python report_generator.py -i data.csv
```

or with AI summary:

```bash
python report_generator.py -i data.csv --use-ai
```

### Custom Output File

To specify a custom output filename:

```bash
python report_generator.py -i data.csv -o my_report.pdf
```

**Note**: Relative paths will be saved in the `reports/` folder. Absolute paths are used as-is.


## Report Contents

Each generated PDF report includes:

1. **Dataset Overview**: Total records, columns, and column type breakdown
2. **Numeric Column Analysis**: For each numeric column:
   - Count, mean, standard deviation
   - Minimum, maximum values
   - Quartiles (25%, 50% median, 75%)
3. **Visual Analysis**:
   - Distribution histograms for numeric columns (up to 6)
   - Frequency bar charts for categorical columns (up to 4)
4. **Data Sample**: First 3 rows displayed in a table format
5. **Executive Summary** (optional, with `--use-ai`):
   - AI-generated insights about data quality
   - Key observations about categorical distributions
   - Notes on skewed numeric columns
   - Overall dataset assessment

## Examples

### Example 1: Quick Report Generation
```bash
# Place your data file (e.g., sales_data.csv) in the directory
python report_generator.py
# Output: reports/sales_data_report_20241101_143022.pdf
```

### Example 2: Full Analysis with AI
```bash
python report_generator.py -i movies_dataset.json --use-ai
# Output: reports/movies_dataset_report_20241101_143022.pdf
# Includes AI summary with key insights
```

### Example 3: Custom Report Name
```bash
python report_generator.py -i sales_data.csv -o monthly_report.pdf
# Output: reports/monthly_report.pdf
```

### Memory issues with large files
- For very large datasets (>1 million rows), consider sampling your data first
- The script limits visualizations to prevent excessive memory usage

## Supported File Formats

- **CSV** (`.csv`): Comma-separated values files
- **Excel** (`.xlsx`, `.xls`): Microsoft Excel spreadsheets
- **JSON** (`.json`): JavaScript Object Notation files (supports both standard and records-oriented JSON)

