import pandas as pd
import streamlit as st

def read_file(uploaded_file):
    """Read an uploaded Excel or CSV file and return a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file {uploaded_file.name}: {e}")
        return None

def highlight_differences(dfs):
    """
    Highlight cell differences across multiple DataFrames.
    Returns a list of styled DataFrames.
    """
    n = len(dfs)
    if n < 2:
        st.warning("Please upload at least two files to compare.")
        return [dfs[0].style] if dfs else []

    # Check if all DataFrames have the same shape
    shapes = [df.shape for df in dfs]
    if len(set(shapes)) != 1:
        st.warning("Uploaded files have different shapes. Comparison may not be accurate.")

    rows, cols = dfs[0].shape

    def cell_differs(i, j):
        values = []
        for df in dfs:
            try:
                values.append(df.iat[i, j])
            except:
                values.append(None)
        # Normalize NaNs so they are comparable
        normalized = [v if pd.notna(v) else '___NaN___' for v in values]
        return len(set(normalized)) > 1

    def style_func(df):
        def highlight_cell(row):
            return [
                'background-color: #ff6961' if cell_differs(row.name, col_idx) else ''
                for col_idx in range(len(row))
            ]
        return df.style.apply(highlight_cell, axis=1)

    styled_dfs = [style_func(df) for df in dfs]
    return styled_dfs
