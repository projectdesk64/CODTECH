import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, Flowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os
import argparse
import io
import json
import logging 
import time
import matplotlib.pyplot as plt
import seaborn as sns 
import requests 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import sys               
import subprocess        
import re 

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generator.log"), 
        logging.StreamHandler() 
    ]
)

# --- AI Summarizer Class (Secure & Robust) ---
class GenerativeSummarizer:
    """
    Uses an AI model (OpenRouter) to generate an executive summary.
    """
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key not found.")
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self.model = "mistralai/mistral-7b-instruct:free" 
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Automated Report Generator"
        }
        
        self.session = requests.Session()
        retries = Retry(
            total=3, 
            backoff_factor=1, 
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate_summary(self, safe_payload_json):
        """
        Calls the AI model with a *small, sanitized* data payload.
        """
        logging.info("Contacting AI for executive summary...")
        
        # --- MODIFIED ---
        # Changed prompt to explicitly ask for HTML <ul> list
        system_prompt = """
        You are a senior data analyst. Your job is to write a high-level, 
        executive summary for a business stakeholder.
        
        You will be given a *sanitized* JSON object describing a dataset. 
        Based *only* on this data, write 3-5 key observations.
        
        - **Data Quality:** Look at 'top_missing_data_columns'. Are there any
          columns with a high percentage of missing values (e.g., 'nan')? 
          Mention them by name.
        - **Key Categories:** Look at 'categorical_top_values'. Are there any
          columns that are heavily dominated by one value (e.g., 'Shipped'
          or 'nan')? This is a key insight.
        - **Distributions:** Look at 'highly_skewed_numeric_columns'. Mention if
          key metrics like 'Sales' or 'Price' are skewed.
        - **Overview:** Start with 1-2 sentences about the dataset's shape.
          
        **CRITICAL: Format your entire response as an HTML unordered list (<ul>...</ul>).**
        Example:
        <ul>
          <li>This dataset contains 150 rows and 5 columns.</li>
          <li>Data quality appears high, with no missing values.</li>
          <li>The 'Species' column is the main categorical feature.</li>
        </ul>
        """
        
        user_prompt = f"""
        Here is the sanitized data profile. Please generate the summary in HTML format.
        
        {safe_payload_json}
        """

        try:
            response = self.session.post(
                url=self.api_url,
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }),
                timeout=30 
            )
            
            response.raise_for_status() 
            
            result = response.json()

            if 'choices' in result and result['choices'] and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                summary_text = result['choices'][0]['message']['content']
                
                # --- MODIFIED ---
                # Clean up the AI output to remove stray tags
                
                # 1. Remove [OUT] tags
                summary_text = re.sub(r'\[/?OUT\]', '', summary_text, flags=re.IGNORECASE)
                
                # 2. Remove markdown list characters if AI adds them
                summary_text = summary_text.replace("*- ", "<li>") 
                
                # 3. Ensure it starts with <ul>
                if not summary_text.strip().startswith("<ul>"):
                    summary_text = "<ul>" + summary_text + "</ul>"
                # --- END MODIFIED SECTION ---

                if summary_text and summary_text.strip():
                    logging.info("AI summary received successfully.")
                    logging.info(f"AI Summary Snippet: {summary_text[:60]}...")
                    return summary_text # No longer replacing \n with <br/>
                else:
                    logging.warning("AI response was empty. This is often an API account/billing issue.")
                    return ("<b>AI Summary Failed:</b> The AI model returned an empty response.<br/><br/>"
                            "<b>This is usually an OpenRouter account issue.</b><br/>"
                            "Please check your account, as you may need to add credits (e.g., $1) to activate the API.")
            else:
                logging.warning(f"AI response format unexpected: {result}")
                return "<b>AI Summary Failed:</b> Received an unexpected response from the API."

        except requests.exceptions.RequestException as e:
            logging.error(f"AI Summary Failed: {e}")
            return ("<b>AI Summary Failed:</b> Could not connect to the API after 3 retries.<br/>"
                    "The report will continue without the summary.")

# --- PDF Helper Functions ---
def header_footer(canvas, doc):
    canvas.saveState()
    header_text = "AUTOMATED DATA PROFILING REPORT"
    canvas.setFont('Helvetica', 10)
    canvas.drawString(doc.leftMargin, doc.height + doc.topMargin + 0.3*inch, header_text)
    footer_text = f"Page {canvas.getPageNumber()}"
    canvas.setFont('Helvetica', 9)
    canvas.drawRightString(doc.width + doc.leftMargin, doc.bottomMargin - 0.2*inch, footer_text)
    canvas.restoreState()

class HLine(Flowable):
    def __init__(self, width=500, color=colors.grey):
        Flowable.__init__(self)
        self.width = width
        self.color = color
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.line(0, 0, self.width, 0)

# --- Report Generator Class ---
class ReportGenerator:
    def __init__(self, data_file, output_file, summarizer=None):
        self.data_file = data_file
        self.output_file = output_file
        self.summarizer = summarizer 
        self.df = None
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.other_cols = []
        
        self.full_analysis = {} 
        self.ai_summary_payload = {} 
        
        sns.set_theme(style="whitegrid", palette="muted")

    def _create_custom_styles(self):
        styles = {}
        styles['title'] = ParagraphStyle('CustomTitle', parent=self.styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1f4788'), spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica-Bold')
        styles['heading'] = ParagraphStyle('CustomHeading', parent=self.styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2d5aa8'), spaceAfter=12, spaceBefore=12, fontName='Helvetica-Bold')
        styles['heading3'] = ParagraphStyle('CustomHeading3', parent=self.styles['Heading3'], fontSize=12, textColor=colors.HexColor('#333333'), spaceAfter=6, spaceBefore=10, fontName='Helvetica-Bold')
        styles['body'] = ParagraphStyle('CustomBody', parent=self.styles['Normal'], fontSize=11, spaceAfter=12, alignment=TA_LEFT, leading=14)
        
        # --- MODIFIED  ---
        # Added left padding for the bullet points
        styles['summary'] = ParagraphStyle(
            'Summary', 
            parent=styles['body'], 
            fontSize=10, 
            leading=14, 
            spaceAfter=15, 
            borderPadding=10, 
            borderColor=colors.HexColor('#AED6F1'), 
            borderWidth=1, 
            backColor=colors.HexColor('#F4F6F6'),
            leftIndent=20 # --- This indents the text for the <ul> list ---
        )
        # --- END MODIFIED SECTION ---
        
        return styles

    def read_data(self):
        """
        Read data from CSV, Excel, or JSON.
        """
        try:
            file_ext = os.path.splitext(self.data_file)[1].lower()
            
            if file_ext == '.csv':
                self.df = pd.read_csv(self.data_file, encoding='latin1')
            elif file_ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.data_file)
            elif file_ext == '.json':
                try:
                    self.df = pd.read_json(self.data_file)
                except ValueError as e:
                    logging.warning(f"Default JSON read failed ({e}). Trying orient='records'...")
                    self.df = pd.read_json(self.data_file, orient='records')
            else:
                logging.error(f"Unsupported file format '{file_ext}'.")
                logging.error("This script supports .csv, .xls, .xlsx, and .json files.")
                return False
            
            logging.info(f"Data loaded successfully from {self.data_file}")
            if self.df.empty:
                logging.error("No data found in file.")
                return False
            
            logging.info(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
            
        except FileNotFoundError:
            logging.error(f"File '{self.data_file}' not found.")
            return False
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            return False

    def analyze_data(self):
        """
        Analyzes data and creates PII-safe payloads.
        """
        if self.df is None: return False
        logging.info("Analyzing data structure...")
        
        MAX_UNIQUE_FOR_CATEGORY = 50 
        
        for col in self.df.columns:
            col_clean = str(col).strip()
            self.df.rename(columns={col: col_clean}, inplace=True)
            col = col_clean

            if pd.api.types.is_object_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                    logging.info(f"Converted object column '{col}' to numeric.")
                except (ValueError, TypeError):
                    pass 

            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            elif self.df[col].nunique() <= MAX_UNIQUE_FOR_CATEGORY:
                self.categorical_cols.append(col)
                self.df[col] = self.df[col].astype(str)
            else:
                self.other_cols.append(col)
        
        logging.info(f"Found {len(self.numeric_cols)} numeric columns.")
        logging.info(f"Found {len(self.categorical_cols)} categorical columns.")
        logging.info(f"Found {len(self.other_cols)} other/text columns.")
        
        stats_df = self.df[self.numeric_cols].describe()
        self.full_analysis = {
            'dataset_shape': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns)
            },
            'numeric_statistics': stats_df.to_dict()
        }
        
        # Build the richer AI payload
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(3)
        skewness = self.df[self.numeric_cols].skew()
        highly_skewed = skewness[(skewness > 1.5) | (skewness < -1.5)].index.tolist()
        top_categorical_values = {}
        for col in self.categorical_cols[:10]:
            top_vals = self.df[col].value_counts(dropna=False).head(3).to_dict()
            top_categorical_values[col] = {
                str(k): v for k, v in top_vals.items()
            }
            
        self.ai_summary_payload = {
            'dataset_shape': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns)
            },
            'column_counts': {
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'other_text': len(self.other_cols)
            },
            'top_missing_data_columns': {
                col: f"{pct:.1f}%" for col, pct in top_missing.items()
            },
            'highly_skewed_numeric_columns': highly_skewed,
            'categorical_top_values': top_categorical_values
        }
        logging.info("Created richer, sanitized AI summary payload.")
        return True

    def _create_title_and_overview(self, analysis):
        """
        Creates the title, source info, and overview table.
        """
        elements = []
        title = Paragraph("AUTOMATED DATA PROFILING REPORT", self.custom_styles['title'])
        elements.append(title)
        
        source_text = f"""
        <b>Data Source:</b> {self.data_file}<br/>
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        elements.append(Paragraph(source_text, self.custom_styles['body']))
        
        elements.append(Paragraph("DATASET OVERVIEW", self.custom_styles['heading']))
        overview_data = [
            ['Metric', 'Value'],
            ['Total Records', f"{analysis['dataset_shape']['total_rows']:,}"],
            ['Total Columns', f"{analysis['dataset_shape']['total_columns']:,}"],
            ['Numeric Columns', len(self.numeric_cols)],
            ['Categorical Columns', len(self.categorical_cols)],
            ['Other/Text/ID Columns', len(self.other_cols)]
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 4.5*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F4F6F6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        elements.append(overview_table)
        elements.append(PageBreak())
        return elements
        
    def _create_statistics_section(self):
        # (This function is unchanged)
        elements = []
        elements.append(Paragraph("NUMERIC COLUMN ANALYSIS", self.custom_styles['heading']))
        
        stats = self.full_analysis['numeric_statistics']
        
        if not self.numeric_cols:
            elements.append(Paragraph("No numeric columns were found in the data.", self.custom_styles['body']))
            return elements

        for col in self.numeric_cols:
            elements.append(Paragraph(f"Column: {col}", self.custom_styles['heading3']))
            col_stats = stats[col]
            data = [
                ['Statistic', 'Value'],
                ['Count', f"{col_stats.get('count', 'N/A'):,.0f}"],
                ['Mean', f"{col_stats.get('mean', 'N/A'):,.2f}"],
                ['Std. Dev.', f"{col_stats.get('std', 'N/A'):,.2f}"],
                ['Min', f"{col_stats.get('min', 'N/A'):,.2f}"],
                ['25% (Q1)', f"{col_stats.get('25%', 'N/A'):,.2f}"],
                ['50% (Median)', f"{col_stats.get('50%', 'N/A'):,.2f}"],
                ['75% (Q3)', f"{col_stats.get('75%', 'N/A'):,.2f}"],
                ['Max', f"{col_stats.get('max', 'N/A'):,.2f}"],
            ]
            
            table = Table(data, colWidths=[1.5*inch, 2.0*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F6')])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.2*inch))
            
        elements.append(PageBreak())
        return elements

    def _create_visual_analysis(self):
        # (This function is unchanged)
        elements = []
        logging.info("Generating visual analysis (with Seaborn)...")
        elements.append(Paragraph("VISUAL ANALYSIS", self.custom_styles['heading']))
        
        # 1. Histograms for Numeric Columns
        elements.append(Paragraph("Numeric Column Distributions", self.custom_styles['heading3']))
        MAX_NUMERIC_CHARTS = 6
        
        if not self.numeric_cols:
            elements.append(Paragraph("No numeric columns to visualize.", self.custom_styles['body']))
        
        for i, col in enumerate(self.numeric_cols):
            if i >= MAX_NUMERIC_CHARTS:
                elements.append(Paragraph(f"<i>(Skipping {len(self.numeric_cols) - i} additional numeric charts)</i>", self.custom_styles['body']))
                break
            try:
                plt.figure(figsize=(8, 4))
                sns.histplot(self.df[col], kde=True, bins=30)
                plt.title(f"Distribution of {col}", fontsize=14, loc='left')
                sns.despine()
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png', dpi=200, bbox_inches='tight')
                plt.close('all') 
                img_data.seek(0)
                elements.append(Image(img_data, width=6.0*inch, height=3.0*inch))
            except Exception as e:
                logging.warning(f"Could not plot histogram for {col}. {e}")

        elements.append(PageBreak())
        
        # 2. Bar Charts for Categorical Columns
        elements.append(Paragraph("Categorical Column Frequencies", self.custom_styles['heading3']))
        MAX_CATEGORICAL_CHARTS = 4
        
        if not self.categorical_cols:
            elements.append(Paragraph("No categorical columns to visualize.", self.custom_styles['body']))
            
        for i, col in enumerate(self.categorical_cols):
            if i >= MAX_CATEGORICAL_CHARTS:
                elements.append(Paragraph(f"<i>(Skipping {len(self.categorical_cols) - i} additional categorical charts)</i>", self.custom_styles['body']))
                break
            try:
                counts = self.df[col].value_counts().head(10)
                plt.figure(figsize=(8, 4))
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f"Top 10 Frequencies for {col}", fontsize=14, loc='left')
                plt.xticks(rotation=45, ha='right')
                sns.despine()
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png', dpi=200, bbox_inches='tight')
                plt.close('all') 
                img_data.seek(0)
                elements.append(Image(img_data, width=6.0*inch, height=3.0*inch))
            except Exception as e:
                logging.warning(f"Could not plot bar chart for {col}. {e}")
                
        return elements

    def _create_data_sample(self):
        # (This function is unchanged)
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("DATA SAMPLE (FIRST 3 ROWS)", self.custom_styles['heading']))
        
        sample_df = self.df.head(3).transpose()
        data = [['Column Name', 'Row 1', 'Row 2', 'Row 3']]
        for col_name, row_data in sample_df.iterrows():
            vals = [str(v)[:50] for v in row_data.values]
            data.append([col_name] + vals)

        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F6')])
        ]))
        elements.append(table)
        return elements

    def _create_summary_section(self):
        # (This function is unchanged from v6.6)
        elements = []
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.custom_styles['heading']))
        
        summary_text = self.summarizer.generate_summary(
            json.dumps(self.ai_summary_payload, indent=2)
        )

        elements.append(Paragraph(summary_text, self.custom_styles['summary']))
        return elements

    def generate_pdf(self):
        # (This function is unchanged from v6.6)
        if self.df is None: return False
        try:
            doc = SimpleDocTemplate(
                self.output_file, pagesize=letter,
                rightMargin=0.75*inch, leftMargin=0.75*inch,
                topMargin=1.0*inch, bottomMargin=0.75*inch
            )
            
            elements = []
            analysis = self.analyze_data()
            if not analysis: 
                logging.error("Data analysis failed.")
                return False
            
            elements.extend(self._create_title_and_overview(self.full_analysis))
            elements.extend(self._create_statistics_section())
            elements.extend(self._create_visual_analysis())
            elements.extend(self._create_data_sample())
            
            if self.summarizer:
                elements.append(PageBreak())
                elements.extend(self._create_summary_section())
            else:
                logging.info("Skipping summary section (AI not enabled).")
            
            doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)
            
            logging.info(f"PDF report generated successfully: {self.output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            plt.close('all') 
            return False

# --- Function to find latest data file ---
def find_latest_data_file():
    """
    Scans the current directory and returns the path of the
    most recently modified data file.
    """
    supported_extensions = ('.csv', '.xlsx', '.xls', '.json')
    data_files = []
    
    for f in os.listdir("."):
        if os.path.isfile(f) and f.endswith(supported_extensions):
            data_files.append(f)
            
    if not data_files:
        return None
        
    latest_file = max(data_files, key=os.path.getmtime)
    return latest_file

# --- Function to auto-open file ---
def open_file(filepath):
    """
    Opens the given file in the default application.
    Cross-platform.
    """
    try:
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin": # macOS
            subprocess.Popen(["open", filepath])
        else: # Linux
            subprocess.Popen(["xdg-open", filepath])
        logging.info(f"Opening report: {filepath}")
    except Exception as e:
        logging.warning(f"Could not auto-open PDF. Error: {e}")
        logging.warning(f"You can find the report here: {filepath}")

# --- Main execution ---
if __name__ == "__main__":
    
    load_dotenv() 
    
    parser = argparse.ArgumentParser(description="Automated Data Profiling Report Generator")
    
    parser.add_argument(
        "-i", "--input", 
        required=False, 
        help="Path to the input file. If omitted, scans for the latest data file."
    )
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        help="Path for the output PDF report. If omitted, a unique name is generated."
    )
    
    parser.add_argument(
        "--use-ai",
        action="store_true", 
        help="Enable AI-powered executive summary. Requires OPENROUTER_API_KEY in .env file."
    )
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("AUTOMATED REPORT GENERATION SYSTEM")
    logging.info("=" * 60)
    
    # Auto-detect logic for input file
    input_file = args.input
    if input_file is None:
        logging.info("No input file specified. Searching current directory...")
        input_file = find_latest_data_file()
        
        if input_file is None:
            logging.error("Could not find any data files (.csv, .xlsx, .xls, .json) in this directory.")
            logging.error("Please add a data file or specify one with the -i flag.")
            exit(1)
        else:
            logging.info(f"Found latest data file: {input_file}")
    else:
        logging.info(f"Using specified input file: {input_file}")
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    logging.info(f"Reports directory: {reports_dir}/")
    
    # Auto-generate logic for output file
    output_file = args.output
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(reports_dir, f"{base_name}_report_{timestamp}.pdf")
        logging.info(f"No output file specified. Using unique name: {output_file}")
    else:
        # If user specified a relative path, put it in reports folder
        # If absolute path, use as-is
        if not os.path.isabs(output_file):
            output_file = os.path.join(reports_dir, output_file)
        logging.info(f"Using specified output file: {output_file}")
        
    summarizer = None
    if args.use_ai:
        api_key = os.getenv("OPENROUTER_API_KEY") 
        if not api_key:
            logging.warning("AI summary requested (--use-ai) but OPENROUTER_API_KEY not found in .env file.")
            logging.warning("The report will be generated *without* an AI summary.")
        else:
            logging.info("OpenRouter API key found. AI summary is ENABLED.")
            summarizer = GenerativeSummarizer(api_key=api_key)
    else:
        logging.info("AI summary is DISABLED. Use --use-ai to enable it.")
    
    logging.info(f"Step 1: Initializing report for '{input_file}'...")
    generator = ReportGenerator(
        data_file=input_file, 
        output_file=output_file,
        summarizer=summarizer
    )
    
    logging.info("Step 2: Reading and processing data...")
    if not generator.read_data():
        logging.error("Halting due to data read error.", exc_info=True)
        exit(1)
        
    logging.info("Step 3: Analyzing data and generating PDF report...")
    
    if generator.generate_pdf():
        logging.info("=" * 60)
        logging.info("REPORT GENERATION COMPLETED SUCCESSFULLY!")
        logging.info(f"Report saved to: {output_file}")
        logging.info("=" * 60)
        
        open_file(output_file)
        
    else:
        logging.error("Report generation failed.", exc_info=True)