import streamlit as st
import pandas as pd
from io import BytesIO
from compare import read_file  # Use your existing read_file function

def link_files():
    st.header("🔗 Link Multiple Excel/CSV Files — Dynamic Value Mapping")

    uploaded_files = st.file_uploader(
        "Upload two or more Excel/CSV files to link columns", 
        type=["xlsx", "xls", "csv"], 
        accept_multiple_files=True,
        key="link_upload"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        dfs = []
        for file in uploaded_files:
            df = read_file(file)
            if df is not None:
                dfs.append(df.fillna(""))  # fill NaNs for smooth editing

        st.markdown("### Uploaded Files")
        for i, f in enumerate(uploaded_files):
            st.write(f"{i+1}. {f.name} (Rows: {dfs[i].shape[0]}, Columns: {dfs[i].shape[1]})")

        source_idx = st.selectbox(
            "Select Source File (where values are edited)", 
            options=range(len(dfs)), 
            format_func=lambda x: uploaded_files[x].name
        )
        source_df = dfs[source_idx]

        source_col = st.selectbox("Select Source Column to Link", options=source_df.columns)

        st.markdown("### Map Target Columns in Other Files")
        mappings = {}
        for i, df in enumerate(dfs):
            if i == source_idx:
                continue
            target_col = st.selectbox(
                f"Map file '{uploaded_files[i].name}' target column for source '{source_col}':",
                options=[None] + list(df.columns),
                key=f"target_col_{i}"
            )
            mappings[i] = target_col

        st.markdown("### Edit Source File Column Values")

        # Extract source column with index preserved
        editable_df = source_df[[source_col]].copy()

        # Use experimental data editor for manual editing
        edited_df = st.data_editor(editable_df, num_rows="dynamic")

        # Button to apply changes across linked files
        if st.button("🔄 Apply Changes to Linked Files"):
            # Update source df column with edited values
            source_df[source_col] = edited_df[source_col]

            # Update linked columns in other files row-wise
            for i, target_col in mappings.items():
                if target_col is None:
                    continue
                target_df = dfs[i]
                min_len = min(len(source_df), len(target_df))
                for idx in range(min_len):
                    target_df.at[idx, target_col] = source_df.at[idx, source_col]

            st.success("✅ Values updated across linked files!")

        st.markdown("---")
        st.markdown("### Updated File Previews")
        for i, df in enumerate(dfs):
            st.subheader(f"File {i+1}: {uploaded_files[i].name}")
            st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Download Updated Files")
        for i, df in enumerate(dfs):
            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)

            st.download_button(
                label=f"📥 Download '{uploaded_files[i].name}'",
                data=buffer,
                file_name=f"updated_{uploaded_files[i].name.split('.')[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.info("Please upload at least two Excel/CSV files to use this linking feature.")
