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
    embeddings = OllamaEmbeddings(model="deepseek-r1:latest")
    # Adjusted parameters for better context retention
    model = OllamaLLM(
        model="deepseek-r1:latest", 
        temperature=0.1,
        num_ctx=8192,  # Increased context window
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9
    )
    return embeddings, model

embeddings, model = load_language_models()

# Enhanced prompt templates with better context preservation
default_template = """
You are a professional data analyst working with tabular data from Excel or CSV files.

DATASET CONTEXT:
{context}

USER QUESTION: {question}

ANALYSIS GUIDELINES:
1. Use ONLY the data provided in the DATASET CONTEXT above
2. If information isn't available in the context, clearly state "Based on the provided data, I cannot determine..."
3. For calculations:
   - Show your work step by step
   - Cite specific column names and values
   - Double-check arithmetic
4. For data quality issues:
   - Point to specific examples from the data
   - Quantify the scope of issues
5. Always reference the actual column names and data structure shown in the context

RESPONSE FORMAT:
- Start with a direct answer to the question
- Follow with supporting details from the data
- End with actionable recommendations if applicable

Your analysis:
"""

summary_template = """
You are a data quality specialist creating a comprehensive analysis report.

DATASET INFORMATION:
{context}

TASK: Create a professional DATA QUALITY REPORT with these sections:

## DATASET OVERVIEW
- Dataset dimensions (rows × columns)
- Column inventory with data types
- Primary purpose/content of the dataset

## DATA QUALITY ASSESSMENT
### Missing Values
- Columns with missing data and percentages
- Impact assessment of missing values

### Data Consistency Issues
- Type mismatches or format inconsistencies
- Duplicate records analysis
- Range/boundary violations

### Data Integrity Concerns
- Outliers and anomalies
- Logical inconsistencies
- Referential integrity issues

## ACTIONABLE RECOMMENDATIONS
### Immediate Fixes (High Priority)
- Critical issues requiring immediate attention
- Specific Excel steps to resolve each issue

### Data Enhancement (Medium Priority)
- Improvements to data structure and quality
- Standardization recommendations

### Prevention Measures (Ongoing)
- Data validation rules to implement
- Quality control procedures

Provide specific, actionable guidance that an Excel user can immediately implement such providing with the VBA in excel and Excel formulas which can help .
"""

defects_template = """
You are an Excel data cleaning specialist. Analyze the following defect report and create a comprehensive repair plan.

DEFECT ANALYSIS:
{context}

Create a DETAILED DATA REPAIR GUIDE with these sections:

## EXECUTIVE SUMMARY
- Total defects identified
- Critical vs. non-critical issues
- Estimated effort required

## CRITICAL DEFECTS (Fix Immediately)
For each critical defect:
### Issue: [Defect Name]
- **Problem**: Detailed explanation
- **Impact**: How it affects data reliability
- **Excel Solution**:
  * Manual steps using Excel interface
  * Formula-based approach: `=FORMULA_HERE`
  * Alternative Power Query/VBA solution if needed
- **Validation**: How to verify the fix worked

## MODERATE DEFECTS (Schedule for Resolution)
[Similar format for medium-priority issues]

## MINOR DEFECTS (Address When Time Permits)
[Simplified fixes for low-priority issues]

## PREVENTION STRATEGY
- Data validation rules to implement
- Excel templates to standardize data entry
- Quality check procedures

Ensure all solutions are specific to Excel and include exact menu paths, formula syntax, and step-by-step instructions.
"""

# Improved function to convert dataframe to text chunks with better context preservation
def dataframe_to_text_chunks(df, chunk_size=2000, chunk_overlap=300, max_rows_per_chunk=75):
    """Convert dataframe to text chunks with enhanced context preservation."""
    df.columns = df.columns.astype(str)
    total_rows = len(df)
    
    chunks = []
    
    # Enhanced schema information with sample data
    schema_info = f"DATASET SCHEMA AND STRUCTURE:\n"
    schema_info += f"Total Dimensions: {total_rows} rows × {len(df.columns)} columns\n\n"
    schema_info += "COLUMN DETAILS:\n"
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Get sample values (non-null)
        sample_values = df[col].dropna().astype(str).head(3).tolist()
        sample_str = f"Examples: {', '.join(sample_values)}" if sample_values else "No valid examples"
        
        schema_info += f"• {col}:\n"
        schema_info += f"  - Type: {dtype}\n"
        schema_info += f"  - Unique values: {unique_count:,}\n"
        schema_info += f"  - Missing: {null_count} ({null_pct:.1f}%)\n"
        schema_info += f"  - {sample_str}\n\n"
    
    schema_doc = Document(
        page_content=schema_info, 
        metadata={"chunk_type": "schema", "priority": "high"}
    )
    chunks.append(schema_doc)
    
    # Enhanced statistics with context
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats_info = "NUMERIC COLUMNS STATISTICAL ANALYSIS:\n\n"
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                stats_info += f"Column: {col}\n"
                stats_info += f"• Count: {int(stats['count']):,} valid values\n"
                stats_info += f"• Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
                stats_info += f"• Average: {stats['mean']:.2f}\n"
                stats_info += f"• Median: {stats['50%']:.2f}\n"
                stats_info += f"• Standard Deviation: {stats['std']:.2f}\n"
                
                # Check for potential issues
                neg_count = (df[col] < 0).sum()
                zero_count = (df[col] == 0).sum()
                if neg_count > 0:
                    stats_info += f"• Negative values: {neg_count}\n"
                if zero_count > 0:
                    stats_info += f"• Zero values: {zero_count}\n"
                
                stats_info += "\n"
            except Exception as e:
                stats_info += f"Column: {col} - Statistics unavailable: {str(e)}\n\n"
        
        stats_doc = Document(
            page_content=stats_info, 
            metadata={"chunk_type": "statistics", "priority": "medium"}
        )
        chunks.append(stats_doc)
    
    # Process data in chunks with better overlap handling
    num_row_chunks = (total_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
    
    for i in range(num_row_chunks):
        start_idx = i * max_rows_per_chunk
        end_idx = min(start_idx + max_rows_per_chunk, total_rows)
        
        # Add overlap from previous chunk if not the first chunk
        if i > 0:
            overlap_start = max(0, start_idx - 5)  # 5 rows overlap
            subset_df = df.iloc[overlap_start:end_idx]
            rows_info = f"DATA CHUNK [{overlap_start+1} to {end_idx}] (includes 5-row overlap):\n"
        else:
            subset_df = df.iloc[start_idx:end_idx]
            rows_info = f"DATA CHUNK [{start_idx+1} to {end_idx}]:\n"
        
        # Convert to CSV with better formatting
        csv_buffer = StringIO()
        subset_df.to_csv(csv_buffer, index=True)  # Include row indices for reference
        csv_text = csv_buffer.getvalue()
        
        chunk_content = rows_info + csv_text
        
        chunk_doc = Document(
            page_content=chunk_content,
            metadata={
                "chunk_type": "data_rows",
                "row_start": start_idx,
                "row_end": end_idx,
                "num_rows": end_idx - start_idx,
                "total_rows": total_rows,
                "priority": "high" if i < 2 else "medium"  # Prioritize first chunks
            }
        )
        chunks.append(chunk_doc)
    
    return chunks

# Enhanced retrieval with better context management
def retrieve_docs(db, query, k=6):
    """Enhanced document retrieval with better context awareness."""
    query_lower = query.lower()
    
    # Always get schema information
    schema_docs = []
    try:
        schema_results = db.similarity_search(query, k=2, filter={"chunk_type": "schema"})
        schema_docs.extend(schema_results)
    except:
        pass
    
    # Determine query type and adjust retrieval strategy
    stats_keywords = ["average", "mean", "maximum", "minimum", "count", "statistics", "sum", "total"]
    structure_keywords = ["column", "field", "schema", "structure", "type", "datatype"]
    
    if any(keyword in query_lower for keyword in stats_keywords):
        # Statistics-focused query
        try:
            stats_docs = db.similarity_search(query, k=2, filter={"chunk_type": "statistics"})
            data_docs = db.similarity_search(query, k=k-len(schema_docs)-len(stats_docs))
            return schema_docs + stats_docs + data_docs
        except:
            pass
    elif any(keyword in query_lower for keyword in structure_keywords):
        # Structure-focused query
        data_docs = db.similarity_search(query, k=k-len(schema_docs))
        return schema_docs + data_docs
    
    # General query - get mixed results with priority weighting
    try:
        # Get high-priority chunks first
        high_priority_docs = db.similarity_search(query, k=k//2, filter={"priority": "high"})
        remaining_k = k - len(high_priority_docs)
        if remaining_k > 0:
            other_docs = db.similarity_search(query, k=remaining_k)
            # Remove duplicates
            seen_content = {doc.page_content for doc in high_priority_docs}
            other_docs = [doc for doc in other_docs if doc.page_content not in seen_content]
            return high_priority_docs + other_docs[:remaining_k]
        return high_priority_docs
    except:
        # Fallback to simple similarity search
        return db.similarity_search(query, k=k)

# Significantly improved query function with better context handling
def question_df(question, documents, prompt_template=None, input_vars=None):
    """Enhanced query function with improved context preservation and error handling."""
    try:
        # Build comprehensive context from documents
        context_parts = []
        
        # Organize documents by type for better context flow
        schema_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "schema"]
        stats_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "statistics"]
        data_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "data_rows"]
        
        # Always include schema for context
        for doc in schema_docs:
            context_parts.append("=== DATASET STRUCTURE ===")
            context_parts.append(doc.page_content)
            context_parts.append("")
        
        # Include statistics if relevant or available
        if stats_docs:
            context_parts.append("=== STATISTICAL SUMMARY ===")
            for doc in stats_docs:
                context_parts.append(doc.page_content)
            context_parts.append("")
        
        # Add data samples with clear separation
        if data_docs:
            context_parts.append("=== DATA SAMPLES ===")
            for i, doc in enumerate(data_docs):
                if i > 0:
                    context_parts.append("--- Next Data Section ---")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Combine all context with clear structure
        full_context = "\n".join(context_parts)
        
        # Use appropriate prompt template
        if prompt_template and input_vars:
            # For custom templates, ensure context is properly included
            if "context" not in input_vars:
                input_vars["context"] = full_context
            elif "{context}" in prompt_template:
                # Replace or supplement existing context
                input_vars["context"] = full_context
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            # Default template with enhanced context
            prompt = ChatPromptTemplate.from_template(default_template)
            input_vars = {"question": question, "context": full_context}
        
        # Create and execute the chain
        chain = prompt | model
        
        # Enhanced retry logic with exponential backoff
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                # Add small delay to prevent overwhelming the model
                if attempt > 0:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                
                response = chain.invoke(input_vars)
                
                # Validate response quality
                if len(response.strip()) < 10:
                    raise ValueError("Response too short, likely incomplete")
                
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries:
                    if any(keyword in error_msg for keyword in ["timeout", "connection", "rate"]):
                        continue  # Retry for network issues
                    elif "context" in error_msg or "token" in error_msg:
                        # Try with reduced context
                        if len(full_context) > 4000:
                            # Truncate context and try again
                            truncated_context = full_context[:4000] + "\n... [Context truncated due to length limits]"
                            input_vars["context"] = truncated_context
                            continue
                
                # If all retries failed, provide helpful error message
                if "context" in error_msg or "token" in error_msg:
                    return ("I apologize, but the dataset is too large for me to process in a single response. "
                           "Please try asking about a specific aspect of your data (e.g., 'analyze column X' or "
                           "'show me the first 10 rows') or consider breaking your question into smaller parts.")
                else:
                    return f"I encountered an error while analyzing your data: {str(e)}. Please try rephrasing your question or check if Ollama is running properly."
                
    except Exception as e:
        return f"Error processing your query: {str(e)}. Please ensure your question is clear and try again."

# Rest of your code remains the same...
# [Include all the other functions like create_vector_store_from_df, get_df_summary, 
# detect_defects_and_format_issues, and ai_chat function here]

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
    st.caption("Powered by Ollama and deepseek-r1:latest • Your local AI data analyst")
    
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