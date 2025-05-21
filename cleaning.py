import streamlit as st
import pandas as pd
import numpy as np
import re

def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### 🧼 Data Cleaning & Manipulation Options")

    # Step 0: Select column to filter rows for targeted cleaning
    st.markdown("#### Select a column to filter rows for cleaning (optional):")
    identifier_col = st.selectbox("Choose a column to filter rows by its values (or leave blank)", options=[None] + list(df.columns))

    # Filter rows based on selected values in the identifier column
    if identifier_col:
        unique_values = df[identifier_col].dropna().unique().tolist()
        selected_values = st.multiselect(
            f"Select values from '{identifier_col}' to apply cleaning (leave empty to select all rows)",
            unique_values
        )
        if selected_values:
            working_df = df[df[identifier_col].isin(selected_values)].copy()
        else:
            working_df = df.copy()
    else:
        working_df = df.copy()

    # 1. Remove Duplicates
    with st.expander("1. Remove Duplicates"):
        if st.checkbox("Remove duplicate rows (on selected rows)"):
            before_count = len(working_df)
            working_df = working_df.drop_duplicates()
            st.success(f"✅ Duplicates removed. Rows before: {before_count}, after: {len(working_df)}")

    # 2. Handle Missing Values
    with st.expander("2. Handle Missing Values"):
        na_action = st.radio("Choose method for missing values", ["None", "Drop rows", "Fill with value", "Fill numeric with mean"])
        if na_action == "Drop rows":
            before_count = len(working_df)
            working_df = working_df.dropna()
            st.success(f"✅ Dropped rows with missing values. Rows before: {before_count}, after: {len(working_df)}")
        elif na_action == "Fill with value":
            fill_value = st.text_input("Value to fill missing cells with:")
            if fill_value:
                working_df = working_df.fillna(fill_value)
                st.success(f"✅ Filled missing cells with '{fill_value}'")
        elif na_action == "Fill numeric with mean":
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    mean_val = working_df[col].mean()
                    working_df[col] = working_df[col].fillna(mean_val)
                st.success(f"✅ Filled missing numeric values with column means")
            else:
                st.info("No numeric columns available for this operation.")

    # 3. Strip Whitespace and Trim Multiple Spaces
    with st.expander("3. Strip Whitespace and Normalize Spaces"):
        if st.checkbox("Strip leading/trailing whitespace from string columns"):
            working_df = working_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            st.success("✅ Whitespace stripped from string columns")
        if st.checkbox("Replace multiple spaces inside strings with single space"):
            def fix_spaces(x):
                if isinstance(x, str):
                    return re.sub(r'\s+', ' ', x)
                else:
                    return x
            working_df = working_df.applymap(fix_spaces)
            st.success("✅ Normalized multiple spaces to single spaces")

    # 4. Remove Special Characters
    with st.expander("4. Remove Special Characters"):
        special_cols = working_df.select_dtypes(include='object').columns.tolist()
        remove_chars_cols = st.multiselect("Select columns to remove special characters from", special_cols)
        if remove_chars_cols:
            def remove_special(x):
                if isinstance(x, str):
                    return re.sub(r'[^A-Za-z0-9\s]', '', x)
                else:
                    return x
            for col in remove_chars_cols:
                working_df[col] = working_df[col].apply(remove_special)
            st.success("✅ Removed special characters from selected columns")

    # 5. Normalize Case
    with st.expander("5. Normalize Case"):
        for col in working_df.select_dtypes(include='object').columns:
            case_option = st.selectbox(f"Change case for column '{col}'", ["None", "lower", "upper", "title"], key=f"case_{col}")
            if case_option == "lower":
                working_df[col] = working_df[col].str.lower()
            elif case_option == "upper":
                working_df[col] = working_df[col].str.upper()
            elif case_option == "title":
                working_df[col] = working_df[col].str.title()

    # 6. Rename Columns
    with st.expander("6. Rename Columns"):
        new_col_names = {}
        for col in working_df.columns:
            new_name = st.text_input(f"Rename column '{col}' to:", value=col, key=f"rename_{col}")
            new_col_names[col] = new_name
        working_df.rename(columns=new_col_names, inplace=True)

    # 7. Change Data Types
    with st.expander("7. Change Data Types"):
        for col in working_df.columns:
            dtype_option = st.selectbox(f"Change data type for column '{col}'", ["No change", "int", "float", "str"], key=f"dtype_{col}")
            try:
                if dtype_option == "int":
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0).astype(int)
                elif dtype_option == "float":
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce').astype(float)
                elif dtype_option == "str":
                    working_df[col] = working_df[col].astype(str)
            except Exception as e:
                st.warning(f"Couldn't convert '{col}' to {dtype_option}: {e}")

    # 8. Add New Column
    with st.expander("8. Add New Column"):
        new_col_name = st.text_input("New column name:")
        new_col_default = st.text_input("Default value for new column (optional):")
        if st.button("➕ Add Column") and new_col_name:
            if new_col_name in working_df.columns:
                st.warning("Column name already exists!")
            else:
                if new_col_default:
                    working_df[new_col_name] = new_col_default
                else:
                    working_df[new_col_name] = np.nan
                st.success(f"✅ Added new column '{new_col_name}'")

    # 9. Delete Columns
    with st.expander("9. Delete Columns"):
        del_cols = st.multiselect("Select columns to delete", working_df.columns.tolist())
        if st.button("🗑️ Delete Selected Columns"):
            if del_cols:
                working_df.drop(columns=del_cols, inplace=True)
                st.success(f"✅ Deleted columns: {', '.join(del_cols)}")
            else:
                st.info("No columns selected for deletion")

    # 10. Sort Data
    with st.expander("10. Sort Data"):
        sort_cols = st.multiselect("Select columns to sort by", working_df.columns.tolist())
        ascending = st.radio("Sort order", options=["Ascending", "Descending"], index=0)
        if st.button("Sort Data"):
            if sort_cols:
                working_df = working_df.sort_values(by=sort_cols, ascending=(ascending == "Ascending"))
                st.success(f"✅ Sorted data by columns: {', '.join(sort_cols)}")
            else:
                st.info("No columns selected to sort by")

    # 11. Filter Rows (Advanced)
    with st.expander("11. Filter Rows (Advanced)"):
        filter_col = st.selectbox("Select column to filter rows", working_df.columns.tolist(), key="filter_col")
        if filter_col:
            dtype = working_df[filter_col].dtype
            if np.issubdtype(dtype, np.number):
                filter_op = st.selectbox("Select filter operator", ["=", ">", "<", ">=", "<=", "!="], key="filter_op_num")
                filter_val = st.number_input("Enter value to filter by", key="filter_val_num")
                if st.button("Apply Numeric Filter"):
                    expr = f"`{filter_col}` {filter_op} @filter_val"
                    working_df = working_df.query(expr)
                    st.success(f"✅ Applied filter: {filter_col} {filter_op} {filter_val}")
            else:
                filter_text = st.text_input("Enter substring to filter by", key="filter_val_str")
                if st.button("Apply Text Filter"):
                    working_df = working_df[working_df[filter_col].str.contains(filter_text, na=False, case=False)]
                    st.success(f"✅ Applied text filter on column '{filter_col}' containing '{filter_text}'")

    # Update original dataframe with changes for selected rows only
    if identifier_col and selected_values:
        df.loc[df[identifier_col].isin(selected_values), working_df.columns] = working_df
    else:
        df = working_df

    return df
