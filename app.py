import streamlit as st
import os
from openai import OpenAI
import json
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Customer Retention Multi-Agent System",
    page_icon="lonely_octopus_logo.png",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    # Try to get API key from Streamlit secrets, .env file, or environment variables
    api_key = None
    
    # Try Streamlit secrets (only if secrets file exists)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        # No secrets file found, that's okay - we'll use .env or environment variable
        pass
    
    # Fall back to environment variable (loaded from .env file via load_dotenv())
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("Please set OPENAI_API_KEY in your .env file, Streamlit secrets, or environment variables")
        st.info("💡 Create a `.env` file in the project root with: `OPENAI_API_KEY=your-api-key-here`")
        st.stop()
    return OpenAI(api_key=api_key)

# Sample retention offers database (in production, this would come from Google Sheets or a database)
RETENTION_OFFERS = [
    {
        "offer_code": "PRICE_DISC_20",
        "offer_name": "20% Discount for 6 Months",
        "description": "20% discount on subscription for the next 6 months",
        "target_reasons": ["too expensive", "price", "cost", "budget", "affordability"]
    },
    {
        "offer_code": "PRICE_DISC_30",
        "offer_name": "30% Discount for 3 Months",
        "description": "30% discount on subscription for the next 3 months",
        "target_reasons": ["too expensive", "price", "cost", "budget", "affordability", "expensive"]
    },
    {
        "offer_code": "FEATURE_UPGRADE",
        "offer_name": "Free Feature Upgrade",
        "description": "Upgrade to premium tier with additional features at no extra cost",
        "target_reasons": ["missing features", "need more features", "limited functionality", "features"]
    },
    {
        "offer_code": "TRIAL_EXTEND",
        "offer_name": "Extended Free Trial",
        "description": "Additional 30 days free trial to explore the platform",
        "target_reasons": ["not sure", "need more time", "trial", "testing", "evaluating"]
    },
    {
        "offer_code": "SUPPORT_PRIORITY",
        "offer_name": "Priority Support Access",
        "description": "Dedicated support team and faster response times",
        "target_reasons": ["support", "customer service", "help", "assistance", "response time"]
    },
    {
        "offer_code": "CUSTOM_SOLUTION",
        "offer_name": "Custom Solution Consultation",
        "description": "Free consultation to create a customized solution for your needs",
        "target_reasons": ["doesn't fit", "not suitable", "custom", "specific needs", "requirements"]
    }
]

def get_offers_database() -> str:
    """Convert offers to a readable format for the AI agent"""
    offers_text = "AVAILABLE RETENTION OFFERS:\n\n"
    for offer in RETENTION_OFFERS:
        offers_text += f"OFFER_CODE: {offer['offer_code']}\n"
        offers_text += f"OFFER_NAME: {offer['offer_name']}\n"
        offers_text += f"DESCRIPTION: {offer['description']}\n"
        offers_text += f"TARGET_REASONS: {', '.join(offer['target_reasons'])}\n"
        offers_text += "\n---\n\n"
    return offers_text

def offer_matcher_agent(client: OpenAI, cancellation_reason: str, offers_db: str) -> Dict:
    """Agent 1: Matches cancellation reasons with retention offers"""
    
    system_message = """ROLE: You are an offer matching specialist for customer retention.

TASK: Match customer cancellation reasons with appropriate retention offers from our available inventory.

INPUT: You will receive:
- A customer's cancellation reason (text explanation)
- Access to our retention offers database

OUTPUT: Provide your response in this exact JSON format:
{
    "OFFER_CODE": "the offer code" OR "NO_MATCH",
    "OFFER_NAME": "the offer name" OR "None",
    "MATCH_REASONING": "1-2 sentences explaining why this offer addresses their concern, or why no match exists"
}

CONSTRAINTS:
- Only recommend offers that directly address the customer's stated cancellation reason
- Never recommend multiple offers - select only the single best match
- If the cancellation reason is vague, return NO_MATCH rather than guessing
- Do not create or modify offer codes - only use existing ones from the database

PROCESS:
1. Analyze the customer's cancellation reason to identify the core issue (price, features, service quality, competitor, etc.)
2. Compare the cancellation reason against each offer's intended purpose
3. Select the offer that most directly resolves their specific concern
4. If no offer addresses their reason (e.g., they're moving countries and we don't service that area), return NO_MATCH"""

    user_message = f"""Please find the best retention offer for this customer.

CANCELLATION REASON: {cancellation_reason}

{offers_db}

Follow your process to analyze this reason and return your recommendation in the specified JSON format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Error in offer matcher agent: {str(e)}")
        return {
            "OFFER_CODE": "NO_MATCH",
            "OFFER_NAME": "None",
            "MATCH_REASONING": f"Error occurred: {str(e)}"
        }

def email_writer_agent(client: OpenAI, matched_offer: str, cancellation_reason: str, customer_email: str) -> str:
    """Agent 2: Writes retention emails based on matched offers"""
    
    system_message = """ROLE: You are an email writer for customer retention.

TASK: Write a brief, friendly email to convince customers to stay using a matched offer.

INPUT: 
- Matched offer (or "NO_MATCH")
- Customer's cancellation reason
- Customer email

OUTPUT:
- If offer is "NO_MATCH": return only "NO_MATCH"
- If offer exists: write a 100-150 word email (no subject line)

CONSTRAINTS:
- Start with the Hey Customer_Name (you can extract it from the email):
- Acknowledge their cancellation reason
- Explain how the offer solves their problem
- End with: "Best regards,\nThe Customer Team"
- Be warm and professional, not pushy
- Include a clear next step for them to take

STRUCTURE:
1. Friendly greeting
2. Acknowledge their concern (1 sentence)
3. Present the offer as a solution (2-3 sentences)
4. Call-to-action (1 sentence)
5. Signature"""

    user_message = f"""Write a retention email:

Matched offer: {matched_offer}
Cancellation reason: {cancellation_reason}
Customer email: {customer_email}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error in email writer agent: {str(e)}")
        return f"Error occurred while generating email: {str(e)}"

def send_team_notification(customer_id: str, customer_email: str, date_cancelled: str) -> tuple[bool, str]:
    """Send notification email to team when no match is found"""
    # Use SENDER_EMAIL as the team email address
    team_email = os.getenv("SENDER_EMAIL")
    
    if not team_email:
        return False, "SENDER_EMAIL not set in .env file"
    
    # Format the email body as specified
    email_body = f"""A customer cancellation was detected but no matching retention offer was found.

Please review manually:

Customer's ID: {customer_id}

Email: {customer_email}

Date Cancelled: {date_cancelled}


Thank you,
"""
    
    return send_email(
        to_email=team_email,
        subject="Manual Review Required - Customer Cancellation",
        body=email_body
    )

def send_email(to_email: str, subject: str, body: str) -> tuple[bool, str]:
    """Send an email using SMTP (supports Gmail and custom domains)"""
    try:
        # Get email configuration from environment variables
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")
        
        if not sender_email or not sender_password:
            return False, "Email configuration missing. Please set SENDER_EMAIL and SENDER_PASSWORD in .env file"
        
        # Validate email format
        if "@" not in to_email or "@" not in sender_email:
            return False, f"Invalid email format. To: {to_email}, From: {sender_email}"
        
        # Determine SMTP settings based on email domain or use provided settings
        if not smtp_server:
            # Auto-detect based on email domain
            if "gmail.com" in sender_email.lower():
                smtp_server = "smtp.gmail.com"
                smtp_port = int(smtp_port) if smtp_port else 587
            elif "outlook.com" in sender_email.lower() or "hotmail.com" in sender_email.lower():
                smtp_server = "smtp-mail.outlook.com"
                smtp_port = int(smtp_port) if smtp_port else 587
            elif "yahoo.com" in sender_email.lower():
                smtp_server = "smtp.mail.yahoo.com"
                smtp_port = int(smtp_port) if smtp_port else 587
            else:
                # For custom domains, require SMTP settings to be set
                return False, f"Custom email domain detected ({sender_email}). Please set SMTP_SERVER and SMTP_PORT in .env file. For Gmail-hosted custom domains, use smtp.gmail.com:587"
        else:
            # Use provided SMTP settings
            smtp_port = int(smtp_port) if smtp_port else 587
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()  # Enable encryption
        server.login(sender_email, sender_password)
        text = msg.as_string()
        
        # Send email and capture any errors
        send_errors = server.sendmail(sender_email, [to_email], text)
        server.quit()
        
        # Check if there were any errors
        if send_errors:
            return False, f"Email sending reported errors: {send_errors}"
        
        return True, f"Email sent successfully to {to_email}! Check your inbox (and spam folder)."
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Authentication failed: {str(e)}. Check your email and password. For Gmail, use an App Password. For custom domains, verify SMTP credentials."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}. Check SMTP_SERVER and SMTP_PORT settings."
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        return False, f"Error sending email ({error_type}): {error_msg}. Verify your email configuration in .env file."

def main():
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'email_sent' not in st.session_state:
        st.session_state.email_sent = False
    if 'team_notification_sent' not in st.session_state:
        st.session_state.team_notification_sent = False
    if 'csv_results' not in st.session_state:
        st.session_state.csv_results = None
    
    # Header with logo and title
    logo_col, title_col = st.columns([0.15, 0.85])
    with logo_col:
        st.image("1.png.webp", width=100)
    with title_col:
        st.title("Customer Retention Multi-Agent System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("This app uses OpenAI's GPT-4o-mini model to automate customer retention.")
        
        # Display offers database
        with st.expander("📋 Available Retention Offers"):
            offers_df = pd.DataFrame(RETENTION_OFFERS)
            st.dataframe(offers_df[['offer_code', 'offer_name', 'description']], use_container_width=True)
        
        # Reset button
        if st.button("🔄 Reset / Start New", use_container_width=True):
            st.session_state.processed_data = None
            st.session_state.email_sent = False
            st.session_state.team_notification_sent = False
            st.session_state.csv_results = None
            st.rerun()
    
    # Tabs for single vs batch processing
    tab1, tab2 = st.tabs(["Single Cancellation", "Batch Processing (CSV)"])
    
    with tab1:
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Input Customer Cancellation")
            
            # Form for customer cancellation input
            with st.form("cancellation_form"):
                customer_id = st.text_input("Customer ID", value="CUST-001")
                customer_email = st.text_input("Email Address", value="customer@example.com")
                cancellation_reason = st.text_area(
                    "Cancellation Reason",
                    height=150,
                    placeholder="e.g., The service is too expensive for my budget..."
                )
                date_cancelled = st.date_input("Date Cancelled")
                
                submitted = st.form_submit_button("Process Cancellation", use_container_width=True)
        
        with col2:
            st.header("Workflow Status")
        
            # Process form submission or use cached data
            if submitted and cancellation_reason:
                # Reset email sent status for new processing
                st.session_state.email_sent = False
                st.session_state.team_notification_sent = False
                # Initialize OpenAI client
                client = get_openai_client()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Get offers database
                status_text.text("Step 1/4: Loading retention offers database...")
                progress_bar.progress(25)
                offers_db = get_offers_database()
                
                # Step 2: Offer Matching Agent
                status_text.text("Step 2/4: Offer Matcher Agent analyzing cancellation reason...")
                progress_bar.progress(50)
                
                with st.spinner("🤖 Offer Matcher Agent is working..."):
                    match_result = offer_matcher_agent(client, cancellation_reason, offers_db)
                
                # Display match results
                st.subheader("🎯 Offer Matching Results")
                match_col1, match_col2 = st.columns([1, 1])
                
                with match_col1:
                    st.metric("Offer Code", match_result.get("OFFER_CODE", "N/A"))
                    st.metric("Offer Name", match_result.get("OFFER_NAME", "N/A"))
                
                with match_col2:
                    st.write("**Match Reasoning:**")
                    st.info(match_result.get("MATCH_REASONING", "N/A"))
                
                # Step 3: Email Writer Agent
                status_text.text("Step 3/4: Email Writer Agent composing retention email...")
                progress_bar.progress(75)
                
                matched_offer_text = f"{match_result.get('OFFER_CODE')} - {match_result.get('OFFER_NAME')}"
                
                with st.spinner("✍️ Email Writer Agent is composing..."):
                    email_content = email_writer_agent(
                        client, 
                        matched_offer_text, 
                        cancellation_reason, 
                        customer_email
                    )
                
                # Step 4: Final Results
                status_text.text("Step 4/4: Processing complete!")
                progress_bar.progress(100)
                
                # Check if match was found
                is_match = match_result.get("OFFER_CODE", "NO_MATCH") != "NO_MATCH" and email_content != "NO_MATCH"
                
                # Store in session state
                st.session_state.processed_data = {
                    'customer_id': customer_id,
                    'customer_email': customer_email,
                    'cancellation_reason': cancellation_reason,
                    'date_cancelled': str(date_cancelled),
                    'match_result': match_result,
                    'email_content': email_content,
                    'is_match': is_match
                }
            
            # Display results from session state
            if st.session_state.processed_data:
                data = st.session_state.processed_data
                customer_id = data['customer_id']
                customer_email = data['customer_email']
                cancellation_reason = data['cancellation_reason']
                date_cancelled = data['date_cancelled']
                match_result = data['match_result']
                email_content = data['email_content']
                is_match = data['is_match']
                
                st.subheader("📧 Generated Retention Email")
                
                if is_match:
                    st.success("✅ Match Found - Email Ready to Send")
                    
                    # Display email preview
                    st.text_area(
                        "Email Preview",
                        value=email_content,
                        height=200,
                        disabled=True,
                        key="email_preview"
                    )
                    
                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📤 Send Email", use_container_width=True, disabled=st.session_state.email_sent):
                            # Show what email we're sending to
                            st.info(f"📧 Sending to: **{customer_email}**")
                            with st.spinner("Sending email..."):
                                success, message = send_email(
                                    to_email=customer_email,
                                    subject="We'd Love to Keep You - Special Retention Offer",
                                    body=email_content
                                )
                                if success:
                                    st.session_state.email_sent = True
                                    st.success(f"{message}")
                                    st.balloons()  # Celebration!
                                    st.rerun()  # Refresh to show updated status
                                else:
                                    st.error(f"{message}")
                                    st.info("💡 Make sure you've set SENDER_EMAIL and SENDER_PASSWORD (Gmail App Password) in your .env file")
                        elif st.session_state.email_sent:
                            st.success(" Email already sent!")
                    
                    with col_btn2:
                        if st.button("📋 Copy Email", use_container_width=True):
                            st.code(email_content, language=None)
                            st.success("Email copied to clipboard!")
                    
                else:
                    st.warning("No Match Found - Requires Manual Review")
                    
                    # Display notification for manual review
                    st.error("**Manual Review Required**")
                    
                    # Send notification email to team (only once)
                    if not st.session_state.team_notification_sent:
                        with st.spinner("Sending notification to team..."):
                            success, message = send_team_notification(
                                customer_id=customer_id,
                                customer_email=customer_email,
                                date_cancelled=date_cancelled
                            )
                            if success:
                                st.session_state.team_notification_sent = True
                                st.success(f"✅ Team notification sent: {message}")
                            else:
                                st.warning(f"⚠️ Could not send team notification: {message}")
                                st.info("💡 Make sure SENDER_EMAIL is set in your .env file")
                    else:
                        st.success("✅ Team notification already sent")
                    
                    # Workflow summary
                    st.markdown("---")
                    st.subheader(" Workflow Summary")
                    
                    workflow_steps = [
                        "Customer cancellation received",
                        "Offer Matcher Agent analyzed cancellation reason",
                        f"No match found",
                        f"Manual review required"
                    ]
                    
                    for step in workflow_steps:
                        st.write(step)
            
            elif submitted:
                st.warning("Please fill in the cancellation reason to proceed.")
            elif not st.session_state.processed_data:
                st.info("Fill in the form on the left to start processing a cancellation.")
    
    with tab2:
        st.header("Batch Process Cancellations from CSV")
        st.info("Upload a CSV file with 'Email' and 'Cancellation Reason' columns to process multiple cancellations at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_uploader")
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Check for required columns
                required_columns = ['Email', 'Cancellation Reason']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info(f"Your CSV should have these columns: {', '.join(required_columns)}")
                else:
                    st.success(f"✅ CSV loaded successfully! Found {len(df)} rows.")
                    st.dataframe(df[required_columns], use_container_width=True)
                    
                    if st.button("🚀 Process All Cancellations", use_container_width=True, type="primary"):
                        # Initialize OpenAI client
                        client = get_openai_client()
                        offers_db = get_offers_database()
                        
                        # Process each row
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            customer_email = str(row['Email']).strip()
                            cancellation_reason = str(row['Cancellation Reason']).strip()
                            
                            # Get customer ID if available, otherwise generate one
                            customer_id = row.get('Customer ID', f"CUST-{idx+1:03d}")
                            if pd.notna(customer_id):
                                customer_id = str(customer_id).strip()
                            else:
                                customer_id = f"CUST-{idx+1:03d}"
                            
                            # Get date cancelled if available
                            date_cancelled = row.get('Date Cancelled', pd.Timestamp.today().strftime('%Y-%m-%d'))
                            if pd.notna(date_cancelled):
                                date_cancelled = str(date_cancelled)
                            else:
                                date_cancelled = pd.Timestamp.today().strftime('%Y-%m-%d')
                            
                            status_text.text(f"Processing {idx+1}/{len(df)}: {customer_email}")
                            progress_bar.progress((idx + 1) / len(df))
                            
                            # Process cancellation
                            with st.spinner(f"Processing {customer_email}..."):
                                # Match offer
                                match_result = offer_matcher_agent(client, cancellation_reason, offers_db)
                                
                                # Generate email
                                matched_offer_text = f"{match_result.get('OFFER_CODE')} - {match_result.get('OFFER_NAME')}"
                                email_content = email_writer_agent(
                                    client,
                                    matched_offer_text,
                                    cancellation_reason,
                                    customer_email
                                )
                                
                                is_match = match_result.get("OFFER_CODE", "NO_MATCH") != "NO_MATCH" and email_content != "NO_MATCH"
                                
                                results.append({
                                    'Customer ID': customer_id,
                                    'Email': customer_email,
                                    'Date Cancelled': date_cancelled,
                                    'Cancellation Reason': cancellation_reason,
                                    'Offer Code': match_result.get('OFFER_CODE', 'NO_MATCH'),
                                    'Offer Name': match_result.get('OFFER_NAME', 'None'),
                                    'Match Found': 'Yes' if is_match else 'No',
                                    'Email Content': email_content,
                                    'Email Sent': False
                                })
                        
                        # Store results in session state
                        st.session_state.csv_results = results
                        st.success(f"✅ Processed {len(results)} cancellations!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        
        # Display results if available
        if st.session_state.csv_results:
            st.markdown("---")
            st.subheader("📊 Processing Results")
            
            results_df = pd.DataFrame(st.session_state.csv_results)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results_df))
            with col2:
                matches = len(results_df[results_df['Match Found'] == 'Yes'])
                st.metric("Matches Found", matches)
            with col3:
                no_matches = len(results_df[results_df['Match Found'] == 'No'])
                st.metric("No Match", no_matches)
            
            # Display results table
            display_df = results_df[['Customer ID', 'Email', 'Offer Code', 'Offer Name', 'Match Found', 'Email Sent']].copy()
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Send emails for matched offers
            matched_results = [r for r in st.session_state.csv_results if r['Match Found'] == 'Yes' and not r['Email Sent']]
            
            if matched_results:
                st.markdown("---")
                st.subheader("📧 Send Retention Emails")
                st.info(f"{len(matched_results)} customers have matched offers ready to send.")
                
                if st.button("📤 Send All Emails", use_container_width=True, type="primary"):
                    success_count = 0
                    fail_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, result in enumerate(matched_results):
                        status_text.text(f"Sending email {idx+1}/{len(matched_results)}: {result['Email']}")
                        progress_bar.progress((idx + 1) / len(matched_results))
                        
                        success, message = send_email(
                            to_email=result['Email'],
                            subject="We'd Love to Keep You - Special Retention Offer",
                            body=result['Email Content']
                        )
                        
                        if success:
                            result['Email Sent'] = True
                            success_count += 1
                        else:
                            fail_count += 1
                    
                    # Update session state
                    st.session_state.csv_results = st.session_state.csv_results
                    
                    st.success(f"✅ Sent {success_count} emails successfully!")
                    if fail_count > 0:
                        st.warning(f"⚠️ {fail_count} emails failed to send.")
                    st.rerun()
            
            # Handle no matches - send team notifications
            no_match_results = [r for r in st.session_state.csv_results if r['Match Found'] == 'No']
            
            if no_match_results:
                st.markdown("---")
                st.subheader("⚠️ No Match Cases")
                st.warning(f"{len(no_match_results)} cancellations had no matching offers.")
                
                # Send team notifications for all no-match cases
                if st.button("📧 Send Team Notifications", use_container_width=True):
                    for result in no_match_results:
                        send_team_notification(
                            customer_id=result['Customer ID'],
                            customer_email=result['Email'],
                            date_cancelled=result['Date Cancelled']
                        )
                    st.success(f"✅ Sent {len(no_match_results)} team notifications!")
            
            # Download results as CSV
            st.markdown("---")
            csv_results = pd.DataFrame(st.session_state.csv_results)
            csv_results_export = csv_results[['Customer ID', 'Email', 'Date Cancelled', 'Cancellation Reason', 
                                            'Offer Code', 'Offer Name', 'Match Found', 'Email Sent']].copy()
            csv_string = csv_results_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_string,
                file_name=f"retention_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Customer Retention Multi-Agent System | Powered by OpenAI GPT-4o-mini</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

