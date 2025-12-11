# Customer Retention Multi-Agent System

A Streamlit application that automatically processes customer cancellations using AI agents to match retention offers and generate personalized emails.

## Features

- **AI-Powered Matching**: Automatically matches cancellation reasons with retention offers
- **Email Generation**: Creates personalized retention emails
- **Email Sending**: Sends emails directly to customers or team notifications
- **Batch Processing**: Process multiple cancellations from CSV files
- **CSV Export**: Download processing results

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
SENDER_EMAIL=your-email@example.com
SENDER_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**For Gmail**: Use an App Password (not your regular password).

### 3. Run the App

```bash
streamlit run app.py
```

Or:

```bash
python3 -m streamlit run app.py
```

Open your browser to `http://localhost:8501`

## Usage

### Single Cancellation

1. Fill in the cancellation form (left side)
2. Click "Process Cancellation"
3. Review the matched offer and generated email
4. Send the email to the customer

### Batch Processing (CSV)

1. Go to the "Batch Processing (CSV)" tab
2. Upload a CSV file with:
   - `Email` (required)
   - `Cancellation Reason` (required)
   - `Customer ID` (optional)
   - `Date Cancelled` (optional)
3. Click "Process All Cancellations"
4. Review results and send emails in bulk

## CSV Format

```csv
Email,Cancellation Reason,Customer ID,Date Cancelled
customer@example.com,The service is too expensive,CUST-001,2025-12-11
customer2@example.com,Missing features I need,CUST-002,2025-12-11
```

## Requirements

- Python 3.8+
- OpenAI API key
- Email account with SMTP access (Gmail recommended)

## License

Free to use and modify.
