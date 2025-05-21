import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from io import StringIO
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Initialize embeddings and model with improved parameters
@st.cache_resource
def load_language_models():
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1)  # Lower temperature for more factual responses
    return embeddings, model

embeddings, model = load_language_models()

# Improved prompt templates
default_template = """
You are a data analysis assistant working with tabular data from Excel or CSV files.

Question: {question}

Table Data:
{context}

Guidelines for answering:
1. Base your answer SOLELY on the data provided in the context.
2. If the required information isn't present in the context, say "I can't answer based on the provided context."
3. When doing calculations:
   - Be precise with numbers
   - Show your calculation steps briefly
   - Explicitly mention which columns you used
4. For aggregation questions (counts, sums, averages):
   - Double-check your math
   - Only include rows that are visible in the context
5. When describing trends or patterns:
   - Cite specific examples from the data
   - Avoid making generalizations beyond what's shown

Data Format Awareness:
- Data is presented as tabular CSV with headers as the first row
- Some columns may have decimal values that need proper handling
- Columns with text data may contain commas inside quoted fields

Your answer:
"""

summary_template = """
### Data Analysis Task
You need to create a professional data quality summary for an Excel/CSV dataset.

### Dataset Information
{context}

### Required Output Format
Create a concise DATA QUALITY REPORT with these sections:

1. DATASET OVERVIEW
   - Total rows and columns
   - Column names and their apparent data types (numeric, text, date, etc.)
   - Brief description of what the dataset appears to contain

2. DATA QUALITY ISSUES
   - Null/missing values (count per column, percentage of total)
   - Potential errors (negative values where inappropriate, outliers)
   - Format inconsistencies (dates, numbers, text)
   - Duplicate records (if detectable)

3. ACTIONABLE RECOMMENDATIONS
   - 3-5 specific Excel actions to clean the data
   - For each suggestion, include the exact Excel feature and how to use it

Keep your report clear, direct, and focused on information that would help someone clean this dataset efficiently.
"""

defects_template = """
You are an Excel data cleaning specialist. Based on the following dataset defect analysis:

{context}

Create a DETAILED DATA REPAIR PLAN with these sections:

1. DEFECT SUMMARY
   - List of all identified defects
   - Priority level for each issue (Critical/Medium/Low)

2. FOR EACH CRITICAL DEFECT:
   - Detailed explanation of the issue
   - Step-by-step Excel fix instructions with:
     * Manual fix procedure using Excel UI
     * Excel formula solution
     * VBA or Power Query solution (if appropriate)
   - Screenshots or detailed descriptions of where to find relevant Excel features

3. FOR MEDIUM AND LOW DEFECTS:
   - Brief explanation and quick fix suggestions

4. PREVENTION PLAN:
   - Data validation rules to prevent future occurrences
   - Excel templates or structures to recommend

Your goal is to provide an Excel user with EVERYTHING they need to fix these issues completely.
"""

# Improved function to convert dataframe to text chunks
def dataframe_to_text_chunks(df, chunk_size=1500, chunk_overlap=200, max_rows_per_chunk=50):
    """Convert dataframe to text chunks with improved handling of tabular structure."""
    # First, ensure we have string column names to avoid issues
    df.columns = df.columns.astype(str)
    
    # Get total number of rows
    total_rows = len(df)
    
    # Calculate how many chunks we need based on max_rows_per_chunk
    num_row_chunks = (total_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
    
    chunks = []
    
    # Create chunk of full schema (column names and types)
    schema_info = "DATASET SCHEMA:\n"
    for col in df.columns:
        col_type = str(df[col].dtype)
        # Try to determine if it's likely a date column
        date_likelihood = "possible date" if any(["date" in col.lower(), "time" in col.lower()]) else ""
        schema_info += f"- {col} ({col_type}) {date_likelihood}\n"
    
    schema_doc = Document(page_content=schema_info, metadata={"chunk_type": "schema"})
    chunks.append(schema_doc)
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats_info = "NUMERIC COLUMN STATISTICS:\n"
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                stats_info += f"Column: {col}\n"
                stats_info += f"- Min: {stats['min']}\n"
                stats_info += f"- Max: {stats['max']}\n"
                stats_info += f"- Mean: {stats['mean']}\n"
                stats_info += f"- Count: {stats['count']}\n"
                stats_info += f"- Missing: {df[col].isna().sum()}\n\n"
            except:
                stats_info += f"Column: {col} - Unable to calculate statistics\n\n"
        
        stats_doc = Document(page_content=stats_info, metadata={"chunk_type": "statistics"})
        chunks.append(stats_doc)
    
    # Process each row chunk
    for i in range(num_row_chunks):
        start_idx = i * max_rows_per_chunk
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        
        # Get the subset of rows
        subset_df = df.iloc[start_idx:end_idx]
        
        # Convert to CSV string but preserve table structure better
        csv_buffer = StringIO()
        subset_df.to_csv(csv_buffer, index=False)
        csv_text = csv_buffer.getvalue()
        
        # Add row indices information to help with context
        rows_info = f"DATA ROWS [{start_idx+1} to {end_idx}]:\n"
        chunk_content = rows_info + csv_text
        
        chunk_doc = Document(
            page_content=chunk_content,
            metadata={
                "chunk_type": "data_rows",
                "row_start": start_idx,
                "row_end": end_idx,
                "num_rows": end_idx - start_idx,
                "total_rows": total_rows
            }
        )
        chunks.append(chunk_doc)
    
    # If there are too many chunks, use a text splitter to further break them down
    if len(chunks) > 15:  # Arbitrary threshold
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # Only apply text splitting to the data row chunks, not schema/stats
        data_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "data_rows"]
        other_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") != "data_rows"]
        
        split_data_chunks = []
        for chunk in data_chunks:
            split_chunks = text_splitter.split_documents([chunk])
            for i, split_chunk in enumerate(split_chunks):
                # Preserve the original metadata and add split information
                split_chunk.metadata.update(chunk.metadata)
                split_chunk.metadata["split_index"] = i
                split_data_chunks.append(split_chunk)
        
        # Combine the split data chunks with the other chunks
        return other_chunks + split_data_chunks
    
    return chunks

# Create vector store from dataframe chunks with improved metadata
def create_vector_store_from_df(df):
    """Create a FAISS vector store from dataframe with improved chunking."""
    chunked_docs = dataframe_to_text_chunks(df)
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    return vector_store

# Enhanced retrieval function with improved ranking
def retrieve_docs(db, query, k=5):
    """Retrieve relevant documents with smarter filtering based on the query."""
    # First, determine if the question is about:
    # 1. Schema/structure (columns, types)
    # 2. Statistics (min, max, average)
    # 3. Specific data rows or filtering
    
    query_lower = query.lower()
    
    # Adjust k based on query type
    k_final = k
    filter_metadata = {}
    
    # Check if the query is about schema or structure
    if any(term in query_lower for term in ["column", "field", "schema", "structure", "data type", "datatype"]):
        # Prioritize schema chunks
        filter_metadata = {"chunk_type": "schema"}
        k_final = min(k, 2)  # Don't need too many schema chunks
    
    # Check if the query is about statistics
    elif any(term in query_lower for term in ["average", "mean", "maximum", "minimum", "count", "statistic"]):
        # Prioritize statistics chunks
        filter_metadata = {"chunk_type": "statistics"}
        k_final = min(k, 3)  # Get a few statistics chunks
    
    # For specific row queries, we'll use the default behavior
    
    # First try with metadata filter if applicable
    if filter_metadata:
        try:
            filtered_docs = db.similarity_search(query, k=k_final, filter=filter_metadata)
            if filtered_docs:
                # If we got results with filter, add some general results too
                general_docs = db.similarity_search(query, k=k-len(filtered_docs))
                return filtered_docs + general_docs
        except:
            # If filtering fails, fall back to regular search
            pass
    
    # Regular similarity search
    return db.similarity_search(query, k=k)

# Improved dataframe summary with more comprehensive checks
def get_df_summary(df):
    """Generate a comprehensive summary of the dataframe."""
    summary_parts = []
    
    # Basic dimensions
    rows, cols = df.shape
    summary_parts.append(f"DATASET DIMENSIONS: {rows} rows × {cols} columns")
    
    # Column names and types
    summary_parts.append("\nCOLUMN INFORMATION:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / rows) * 100 if rows > 0 else 0
        
        # Try to detect if it's a potential date column
        date_hint = ""
        if "date" in col.lower() or "time" in col.lower():
            sample_values = df[col].dropna().astype(str).tolist()[:3]
            date_hint = f"(Possible date column. Sample values: {', '.join(sample_values[:3])})"
        
        col_info = f"- {col}: {dtype}, {unique_count} unique values, {null_count} nulls ({null_pct:.1f}%) {date_hint}"
        summary_parts.append(col_info)
    
    # Missing values summary
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        summary_parts.append("\nMISSING VALUES:")
        for col, count in null_cols.items():
            summary_parts.append(f"- {col}: {count} nulls ({(count/rows)*100:.1f}%)")
    else:
        summary_parts.append("\nMISSING VALUES: None detected")
    
    # Potential data issues
    summary_parts.append("\nPOTENTIAL DATA ISSUES:")
    
    # Check for negative values in numeric columns
    neg_issues = []
    for col in df.select_dtypes(include=["number"]).columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            neg_issues.append(f"- {col}: {neg_count} negative values")
    
    if neg_issues:
        summary_parts.extend(neg_issues)
    else:
        summary_parts.append("- No negative values detected in numeric columns")
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    summary_parts.append(f"- Duplicate rows: {dup_count} ({(dup_count/rows)*100:.1f}% of data)")
    
    # Check for potential outliers in numeric columns
    outlier_issues = []
    for col in df.select_dtypes(include=["number"]).columns:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outlier_issues.append(f"- {col}: {outlier_count} potential outliers")
        except:
            continue
    
    if outlier_issues:
        summary_parts.append("\nPOTENTIAL OUTLIERS:")
        summary_parts.extend(outlier_issues)
    
    return "\n".join(summary_parts)

# Enhanced defect detection
def detect_defects_and_format_issues(df):
    """Detect comprehensive data defects and format issues."""
    defects = {}
    rows, cols = df.shape
    
    # Check for negative values in numeric columns
    for col in df.select_dtypes(include=["number"]).columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            defects[f"Negative values in '{col}'"] = {
                "description": f"Found {neg_count} negative values in column '{col}'.",
                "affected_rows": neg_count,
                "percentage": f"{(neg_count/rows)*100:.1f}%",
                "severity": "Medium" if neg_count > rows * 0.01 else "Low",
                "fix_suggestion": f"Use Excel's conditional formatting to highlight cells < 0 in column '{col}', then review if these should be absolute values or are data entry errors.",
                "excel_formula": f"=IF({col}2<0,ABS({col}2),{col}2)",
                "excel_conditional_format": f"Format cells where: Cell Value < 0"
            }
    
    # Check for missing values
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    for col, count in null_cols.items():
        severity = "High" if count > rows * 0.2 else "Medium" if count > rows * 0.05 else "Low"
        defects[f"Missing values in '{col}'"] = {
            "description": f"Found {count} missing values ({(count/rows)*100:.1f}%) in column '{col}'.",
            "affected_rows": count,
            "percentage": f"{(count/rows)*100:.1f}%", 
            "severity": severity,
            "fix_suggestion": f"Use Excel's Filter to show only blank cells in column '{col}'. Then decide whether to use a default value, interpolate, or leave blank.",
            "excel_formula": f'=IF(ISBLANK({col}2),"MISSING",{col}2)',
            "excel_conditional_format": "Format cells that contain: Blanks"
        }
    
    # Check for inconsistent data types
    for col in df.columns:
        data_type = df[col].dtype
        
        # Check if date columns have consistent formats
        if "date" in str(col).lower() or "time" in str(col).lower():
            # Try to convert to datetime
            try:
                # Get sample of non-null values
                sample = df[col].dropna().astype(str).tolist()[:10]
                
                # Check if there are multiple date formats
                date_formats = set()
                for val in sample:
                    # Simple pattern matching for date formats
                    if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', val):
                        date_formats.add("MM/DD/YYYY or DD/MM/YYYY")
                    elif re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val):
                        date_formats.add("YYYY/MM/DD")
                    elif re.match(r'[A-Za-z]{3,} \d{1,2},? \d{4}', val):
                        date_formats.add("Month DD, YYYY")
                
                if len(date_formats) > 1:
                    defects[f"Inconsistent date formats in '{col}'"] = {
                        "description": f"Multiple date formats detected in column '{col}': {', '.join(date_formats)}",
                        "affected_rows": "Unknown",
                        "percentage": "Unknown",
                        "severity": "High",
                        "fix_suggestion": f"Use Excel's Text to Columns with Date format selected, or create a helper column with the DATEVALUE function to standardize dates.",
                        "excel_formula": f'=DATEVALUE({col}2)',
                        "excel_conditional_format": "Custom format: highlight cells that can't convert to dates"
                    }
            except:
                pass
        
        # For numeric columns, check for text mixed in
        if df[col].dtype in ['int64', 'float64']:
            # Try converting to numeric and check for errors
            try:
                errors = pd.to_numeric(df[col], errors='coerce').isna() & (~df[col].isna())
                error_count = errors.sum()
                if error_count > 0:
                    defects[f"Text values in numeric column '{col}'"] = {
                        "description": f"Found {error_count} text values in numeric column '{col}'.",
                        "affected_rows": error_count,
                        "percentage": f"{(error_count/rows)*100:.1f}%",
                        "severity": "High",
                        "fix_suggestion": f"Use Excel's Error Checking feature to identify cells with text in numeric column '{col}'. Then correct or remove the text values.",
                        "excel_formula": '=IFERROR(VALUE(' + col + '2),"Text value detected")',
                        "excel_conditional_format": "Use 'Format only cells with: Formula is: =ISTEXT(' + col + '2)"
                    }
            except:
                pass
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        defects["Duplicate rows"] = {
            "description": f"Found {dup_count} duplicate rows ({(dup_count/rows)*100:.1f}% of data).",
            "affected_rows": dup_count,
            "percentage": f"{(dup_count/rows)*100:.1f}%",
            "severity": "Medium",
            "fix_suggestion": "Use Excel's Remove Duplicates feature (Data tab > Remove Duplicates) to identify and eliminate duplicate rows.",
            "excel_formula": "N/A - Use built-in Remove Duplicates feature",
            "excel_conditional_format": "Use conditional formatting with a COUNTIFS formula to highlight duplicates"
        }
    
    # Check for outliers in numeric columns
    for col in df.select_dtypes(include=["number"]).columns:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0 and outlier_count < rows * 0.1:  # Ignore if too many "outliers"
                defects[f"Outliers in '{col}'"] = {
                    "description": f"Found {outlier_count} outliers in column '{col}' (values outside {lower_bound:.2f} to {upper_bound:.2f}).",
                    "affected_rows": outlier_count,
                    "percentage": f"{(outlier_count/rows)*100:.1f}%",
                    "severity": "Low",
                    "fix_suggestion": f"Use Excel's conditional formatting to highlight values outside the range {lower_bound:.2f} to {upper_bound:.2f} in column '{col}', then verify if they are legitimate or errors.",
                    "excel_formula": f"=OR({col}2<{lower_bound},{col}2>{upper_bound})",
                    "excel_conditional_format": f"Format cells where: Cell Value < {lower_bound} OR Cell Value > {upper_bound}"
                }
        except:
            continue
    
    # Format the defects into a string for the AI model
    if not defects:
        return "No data defects or format issues detected."
    
    # Format into a detailed report
    defect_lines = ["DETAILED DEFECTS REPORT:", ""]
    
    # Sort defects by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    sorted_defects = sorted(defects.items(), key=lambda x: severity_order.get(x[1]["severity"], 999))
    
    for defect_name, details in sorted_defects:
        defect_lines.append(f"DEFECT: {defect_name}")
        defect_lines.append(f"Severity: {details['severity']}")
        defect_lines.append(f"Description: {details['description']}")
        defect_lines.append(f"Affected rows: {details['affected_rows']} ({details['percentage']})")
        defect_lines.append(f"Fix suggestion: {details['fix_suggestion']}")
        defect_lines.append(f"Excel formula: {details['excel_formula']}")
        if "excel_conditional_format" in details:
            defect_lines.append(f"Conditional format: {details['excel_conditional_format']}")
        defect_lines.append("")  # Empty line between defects
    
    return "\n".join(defect_lines)

# Enhanced query function with error handling
def question_df(question, documents, prompt_template=None, input_vars=None):
    """Query the Ollama model with improved context and error handling."""
    try:
        # Prepare context based on document type
        context_parts = []
        
        # Process documents by type to provide better context
        schema_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "schema"]
        stats_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "statistics"]
        data_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "data_rows"]
        
        # Always include schema if available
        if schema_docs:
            context_parts.append(schema_docs[0].page_content)
        
        # Include stats if available and relevant to the question
        stats_keywords = ["average", "mean", "maximum", "minimum", "count", "statistics", "range"]
        if stats_docs and any(keyword in question.lower() for keyword in stats_keywords):
            context_parts.append(stats_docs[0].page_content)
        
        # Add data rows
        for doc in data_docs:
            context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)
        
        # Use the specified prompt or default
        if prompt_template and input_vars:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            prompt = ChatPromptTemplate.from_template(default_template)
            input_vars = {"question": question, "context": context}
        
        # Execute the chain
        chain = prompt | model
        
        # Handle potential timeout issues with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return chain.invoke(input_vars)
            except Exception as e:
                if "timeout" in str(e).lower() and attempt < max_retries:
                    time.sleep(2)  # Wait before retry
                    continue
                raise
                
    except Exception as e:
        return f"Error processing your query: {str(e)}. Please try again with a different question or check if Ollama is running correctly."

# Define enhanced FAQ questions
FAQ_QUESTIONS = [
    "📊 Dataset Summary",
    "🧩 Missing Values Analysis",
    "🔍 Numeric Data Issues",
    "🔧 Excel Cleaning Steps",
    "📉 Data Quality Score",
    "📈 Distribution Overview"
]

# Enhanced Streamlit app
def ai_chat():
    
    st.title("🤖 Excel & CSV Data Assistant")
    st.caption("Powered by Ollama and deepseek-r1:1.5b • Your local AI data analyst")
    
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "faq_response" not in st.session_state:
        st.session_state.faq_response = None
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = []
    if "file_names" not in st.session_state:
        st.session_state.file_names = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        
    # Sidebar for history and settings
    with st.sidebar:
        st.header("📝 Options")
        
        if st.button("📜 Show/Hide Chat History"):
            st.session_state.show_history = not st.session_state.show_history
            
        if st.session_state.show_history and st.session_state.chat_history:
            st.subheader("🗂️ Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 items
                with st.expander(f"Q{i+1}: {question[:50]}..."):
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
        
        st.markdown("---")
        st.caption("Built with Streamlit + LangChain")
    
    # File upload area
    uploaded_files = st.file_uploader(
        "📄 Upload Excel or CSV Files",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload one or more Excel/CSV files for analysis"
    )
    
    # Process uploaded files
    if uploaded_files and not st.session_state.data_loaded:
        with st.spinner("📊 Processing your data files..."):
            st.session_state.dataframes = []
            st.session_state.file_names = []
            
            for file in uploaded_files:
                try:
                    if file.name.endswith(".csv"):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file, sheet_name=0)
                    
                    # Basic data cleaning
                    # Convert column names to strings
                    df.columns = df.columns.astype(str)
                    
                    # Remove completely empty columns and rows
                    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
                    
                    st.session_state.dataframes.append(df)
                    st.session_state.file_names.append(file.name)
                except Exception as e:
                    st.error(f"❌ Failed to read '{file.name}': {str(e)}")
            
            st.session_state.data_loaded = True
    
    # Display file options and dataset preview if files are loaded
    if st.session_state.data_loaded and st.session_state.dataframes:
        # Options for data processing
        col1, col2 = st.columns(2)
        
        with col1:
            combine = st.checkbox("🔗 Combine all files into one dataset", value=len(st.session_state.dataframes) > 1)
        
        with col2:
            if not combine and len(st.session_state.file_names) > 1:
                file_choice = st.selectbox(
                    "Select a file to analyze", 
                    options=st.session_state.file_names,
                    index=0
                )
                idx = st.session_state.file_names.index(file_choice)
                selected_df = st.session_state.dataframes[idx]
                st.session_state.current_df = selected_df
            else:
                # If only one file or combining is selected
                if len(st.session_state.dataframes) == 1:
                    st.session_state.current_df = st.session_state.dataframes[0]
                else:
                    st.session_state.current_df = pd.concat(st.session_state.dataframes, ignore_index=True)
        
        # Preview the data
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(st.session_state.current_df.head(10), use_container_width=True)
            
            # Show basic dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(st.session_state.current_df):,}")
            with col2:
                st.metric("Columns", f"{len(st.session_state.current_df.columns):,}")
            with col3:
                missing = st.session_state.current_df.isna().sum().sum()
                st.metric("Missing Values", f"{missing:,}")
        
        # Create vector store for the current dataframe if not already created
        if st.session_state.vector_store is None:
            with st.spinner("🔍 Building vector index for your data..."):
                try:
                    st.session_state.vector_store = create_vector_store_from_df(st.session_state.current_df)
                    st.success("✅ Vector index created successfully!")
                except Exception as e:
                    st.error(f"❌ Error creating vector index: {str(e)}")
                    st.session_state.vector_store = None
        
        # Main interaction area
        st.markdown("## 💬 Ask questions about your dataset")
        
        # Chat interface
        user_input = st.text_input(
            "Enter your question about the data:",
            placeholder="E.g., What's the average value in column X? or How many rows have missing values?",
            key="user_question"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_button = st.button("🚀 Ask AI", type="primary", use_container_width=True)
        
        with col2:
            if st.session_state.vector_store is not None:
                clear_button = st.button("🧹 Clear", use_container_width=True)
                if clear_button:
                    st.session_state.faq_response = None
        
        # Process user query
        if ask_button and user_input.strip() and st.session_state.vector_store is not None:
            with st.spinner("🧠 Analyzing your data..."):
                try:
                    # Retrieve relevant chunks from the vector store
                    docs = retrieve_docs(st.session_state.vector_store, user_input, k=5)
                    
                    # Get response from the model
                    response = question_df(user_input, docs)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_input, response))
                    st.session_state.faq_response = response
                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")
        
        # Display response
        if st.session_state.faq_response:
            st.markdown("### 🤖 Answer")
            st.markdown(st.session_state.faq_response)
            st.markdown("---")
        
        # Quick analysis buttons
        st.markdown("## 🔍 Quick Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Dataset Summary", use_container_width=True):
                with st.spinner("Generating comprehensive dataset summary..."):
                    try:
                        # Generate detailed summary of the dataframe
                        summary_context = get_df_summary(st.session_state.current_df)
                        summary_docs = [Document(page_content=summary_context)]
                        
                        # Query the model with the summary context
                        response = question_df(
                            "Create a professional data quality summary for this dataset.",
                            summary_docs,
                            summary_template,
                            {"context": summary_context}
                        )
                        
                        # Update the response state
                        st.session_state.faq_response = response
                        st.session_state.chat_history.append(("Generate Dataset Summary", response))
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
        
        with col2:
            if st.button("🚨 Data Quality Issues", use_container_width=True):
                with st.spinner("Analyzing data quality and defects..."):
                    try:
                        # Detect defects in the dataframe
                        defects_report = detect_defects_and_format_issues(st.session_state.current_df)
                        defects_docs = [Document(page_content=defects_report)]
                        
                        # Query the model with the defects context
                        response = question_df(
                            "Create a detailed data repair plan for the defects in this dataset.",
                            defects_docs,
                            defects_template,
                            {"context": defects_report}
                        )
                        
                        # Update the response state
                        st.session_state.faq_response = response
                        st.session_state.chat_history.append(("Analyze Data Quality Issues", response))
                    except Exception as e:
                        st.error(f"❌ Error analyzing defects: {str(e)}")
        
        # FAQ buttons section
        st.markdown("## 📚 Common Data Questions")
        
        # Display FAQ buttons in a grid
        cols = st.columns(3)
        for i, question in enumerate(FAQ_QUESTIONS):
            if cols[i % 3].button(question, key=f"faq_{i}", use_container_width=True):
                with st.spinner(f"Answering: {question}..."):
                    try:
                        # Map FAQ buttons to appropriate actions
                        if "Summary" in question:
                            # Generate dataset summary
                            summary_context = get_df_summary(st.session_state.current_df)
                            docs = [Document(page_content=summary_context)]
                            response = question_df(
                                "Provide a concise summary of this dataset with key insights.",
                                docs,
                                summary_template,
                                {"context": summary_context}
                            )
                        elif "Missing" in question:
                            # Analyze missing values
                            missing_report = ""
                            for col in st.session_state.current_df.columns:
                                missing_count = st.session_state.current_df[col].isna().sum()
                                if missing_count > 0:
                                    missing_pct = (missing_count / len(st.session_state.current_df)) * 100
                                    missing_report += f"- Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)\n"
                            
                            if not missing_report:
                                missing_report = "No missing values found in the dataset."
                            
                            docs = [Document(page_content=missing_report)]
                            response = question_df(
                                "Analyze the missing values in this dataset and suggest how to handle them.",
                                docs
                            )
                        elif "Numeric" in question:
                            # Analyze numeric columns
                            numeric_cols = st.session_state.current_df.select_dtypes(include=['number']).columns
                            numeric_report = "Numeric columns analysis:\n\n"
                            
                            for col in numeric_cols:
                                try:
                                    stats = st.session_state.current_df[col].describe()
                                    numeric_report += f"Column: {col}\n"
                                    numeric_report += f"- Min: {stats['min']}\n"
                                    numeric_report += f"- Max: {stats['max']}\n"
                                    numeric_report += f"- Mean: {stats['mean']}\n"
                                    numeric_report += f"- Median: {stats['50%']}\n"
                                    numeric_report += f"- Negative values: {(st.session_state.current_df[col] < 0).sum()}\n\n"
                                except:
                                    numeric_report += f"Column: {col} - Unable to calculate statistics\n\n"
                            
                            docs = [Document(page_content=numeric_report)]
                            response = question_df(
                                "Analyze the numeric data in this dataset and identify any issues or patterns.",
                                docs
                            )
                        elif "Cleaning" in question:
                            # Generate cleaning recommendations
                            defects_report = detect_defects_and_format_issues(st.session_state.current_df)
                            docs = [Document(page_content=defects_report)]
                            response = question_df(
                                "Provide step-by-step Excel procedures to clean this dataset.",
                                docs,
                                defects_template,
                                {"context": defects_report}
                            )
                        elif "Quality Score" in question:
                            # Generate data quality score
                            df = st.session_state.current_df
                            total_cells = df.shape[0] * df.shape[1]
                            missing_cells = df.isna().sum().sum()
                            missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
                            
                            # Check for duplicate rows
                            duplicate_rows = df.duplicated().sum()
                            duplicate_pct = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
                            
                            # Check for negative values in numeric columns
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            neg_values = sum((df[col] < 0).sum() for col in numeric_cols)
                            
                            # Generate overall quality report
                            quality_report = f"""
                            Data Quality Analysis:
                            - Total rows: {len(df)}
                            - Total columns: {len(df.columns)}
                            - Missing values: {missing_cells} ({missing_pct:.1f}% of all cells)
                            - Duplicate rows: {duplicate_rows} ({duplicate_pct:.1f}% of all rows)
                            - Negative values in numeric columns: {neg_values}
                            """
                            
                            docs = [Document(page_content=quality_report)]
                            response = question_df(
                                "Calculate a data quality score for this dataset and explain the rating.",
                                docs
                            )
                        elif "Distribution" in question:
                            # Generate distribution overview
                            distribution_report = "Column Distribution Overview:\n\n"
                            
                            # Check categorical columns
                            categorical_cols = st.session_state.current_df.select_dtypes(include=['object']).columns
                            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                                try:
                                    value_counts = st.session_state.current_df[col].value_counts().head(5)
                                    distribution_report += f"Column '{col}' (top 5 values):\n"
                                    for val, count in value_counts.items():
                                        pct = (count / len(st.session_state.current_df)) * 100
                                        distribution_report += f"- {val}: {count} ({pct:.1f}%)\n"
                                    distribution_report += "\n"
                                except:
                                    distribution_report += f"Column '{col}': Unable to calculate distribution\n\n"
                            
                            # Check numeric columns
                            numeric_cols = st.session_state.current_df.select_dtypes(include=['number']).columns
                            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                                try:
                                    distribution_report += f"Column '{col}' (numeric distribution):\n"
                                    bins = [0, 25, 50, 75, 100]
                                    quantile_values = [st.session_state.current_df[col].quantile(q/100) for q in bins]
                                    
                                    for i in range(len(bins)-1):
                                        lower = quantile_values[i]
                                        upper = quantile_values[i+1]
                                        count = ((st.session_state.current_df[col] >= lower) & 
                                                (st.session_state.current_df[col] <= upper)).sum()
                                        pct = (count / len(st.session_state.current_df)) * 100
                                        distribution_report += f"- {bins[i]}%-{bins[i+1]}% range ({lower:.2f} to {upper:.2f}): {count} rows ({pct:.1f}%)\n"
                                    distribution_report += "\n"
                                except:
                                    distribution_report += f"Column '{col}': Unable to calculate distribution\n\n"
                            
                            docs = [Document(page_content=distribution_report)]
                            response = question_df(
                                "Analyze the distribution of values in this dataset.",
                                docs
                            )
                        else:
                            # Default case - retrieve from vector store
                            docs = retrieve_docs(st.session_state.vector_store, question, k=4)
                            response = question_df(question, docs)
                            
                        # Update the response state
                        st.session_state.faq_response = response
                        st.session_state.chat_history.append((question, response))
                    except Exception as e:
                        st.error(f"❌ Error answering FAQ: {str(e)}")
    else:
        st.info("👆 Please upload one or more Excel or CSV files to start analyzing your data.")
        
        # Show feature highlights when no data is loaded
        st.markdown("""
        ### 🌟 Feature Highlights
        
        - **Natural Language Queries**: Ask questions about your data in plain English
        - **Smart Data Analysis**: Get insights on data quality, patterns, and issues
        - **Excel Integration**: Receive Excel-specific formulas and steps to fix data issues
        - **Privacy Focused**: All processing happens locally - your data never leaves your computer
        
        ### 📝 Example Questions
        
        Once you upload data, you can ask questions like:
        - "What's the average value in column X?"
        - "How many rows have missing values?"
        - "Find potential duplicates in the data"
        - "Identify rows where value in column A is greater than 100"
        - "What's the distribution of values in column B?"
        """)

if __name__ == "__main__":
    ai_chat()