import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from io import BytesIO
from fuzzywuzzy import process

def analyze_consecutive_ids(df, column='Media_ID'):
    media_ids = df[column].dropna().astype(str)
    grouped = defaultdict(list)

    for media_id in media_ids:
        match = re.match(r'(T\d+)(\d{2})([A-Z]+)', media_id)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            suffix = match.group(3)
            key = f"{prefix}{suffix}"
            grouped[key].append(number)

    consecutive_counts = {}
    for key, numbers in grouped.items():
        sorted_nums = sorted(numbers)
        count = 1
        sub_count = 1
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i - 1] + 1:
                sub_count += 1
            else:
                consecutive_counts[f"{key}_{count}"] = sub_count
                count += 1
                sub_count = 1
        consecutive_counts[f"{key}_{count}"] = sub_count

    consec_df = pd.DataFrame(list(consecutive_counts.items()), columns=["Group", "Count"])
    consec_df.sort_values(by="Group", inplace=True)
    consec_df.reset_index(drop=True, inplace=True)
    return consec_df

def summarize_media_by_area(df, media_col='Media_ID', area_col='Area Name / Block'):
    df = df.dropna(subset=[media_col, area_col])
    summary = df.groupby(area_col)[media_col].nunique().reset_index()
    summary.rename(columns={media_col: 'No. of Media'}, inplace=True)
    summary['No. of Media'] = summary['No. of Media'].astype(int)
    summary.reset_index(drop=True, inplace=True)
    return summary

def match_area_name(area_name, area_list, threshold=80, use_fuzzy=False):
    if not isinstance(area_name, str):
        return None

    for area in area_list:
        if area_name.strip().lower() == area.strip().lower():
            return area

    if use_fuzzy:
        match, score = process.extractOne(area_name, area_list)
        if score >= threshold:
            return match

    return None

def analyze_app():
    st.title("🔍 Analyze Media_ID Consecutive Sequences and Area Summary")

    uploaded_files = st.file_uploader(
        "Upload one or more Excel files (First Input)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="analyze_upload_multiple"
    )

    if uploaded_files and len(uploaded_files) > 0:
        combined_df = pd.DataFrame()
        sheet_name = None

        # Read all uploaded files and combine their sheets with the same name or first sheet
        for file in uploaded_files:
            try:
                sheets = pd.read_excel(file, None)
                if sheet_name is None:
                    # Just pick the first sheet name from the first file
                    sheet_name = list(sheets.keys())[0]
                # Try to get that sheet name or fallback to first sheet
                df_file = sheets.get(sheet_name, None)
                if df_file is None:
                    df_file = sheets[list(sheets.keys())[0]]
                combined_df = pd.concat([combined_df, df_file], ignore_index=True)
            except Exception as e:
                st.error(f"Error reading {file.name}: {str(e)}")

        if combined_df.empty:
            st.warning("No data loaded from the uploaded files.")
            return

        st.subheader("📄 Preview of Combined Data from All Uploaded Files")
        st.dataframe(combined_df.head(), use_container_width=True)

        if 'Media_ID' not in combined_df.columns or 'Area Name / Block' not in combined_df.columns:
            st.warning("The combined data must contain 'Media_ID' and 'Area Name / Block' columns.")
            return

        if st.checkbox("Show consecutive Media_ID analysis"):
            result_df = analyze_consecutive_ids(combined_df)
            st.subheader("📊 Consecutive Media_ID Analysis")
            st.dataframe(result_df, use_container_width=True)

        summary_df = summarize_media_by_area(combined_df)
        st.subheader("📋 Summary: Number of Media per Area Name / Block")
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🔄 Upload second Excel for Area comparison and update counts")
        uploaded_file_2 = st.file_uploader(
            "Upload second Excel file",
            type=["xlsx", "xls"],
            key="analyze_upload_2"
        )

        if uploaded_file_2:
            sheets_2 = pd.read_excel(uploaded_file_2, None)
            sheet_names_2 = list(sheets_2.keys())
            selected_sheet_2 = st.selectbox("Select sheet from second file", sheet_names_2, key='sheet2')

            df2 = pd.read_excel(uploaded_file_2, sheet_name=selected_sheet_2, skiprows=2)
            df2.columns = df2.columns.str.strip()
            cols_lower = [c.lower() for c in df2.columns]

            if 'area name / block' not in cols_lower or 'no. of media' not in cols_lower:
                st.warning("The second sheet must contain 'Area Name / Block' and 'No. of Media' columns.")
                return

            area_col = df2.columns[cols_lower.index('area name / block')]
            media_col = df2.columns[cols_lower.index('no. of media')]

            st.subheader("📄 Preview of Second Uploaded Sheet (Starting from 3rd Row)")
            st.dataframe(df2.head(), use_container_width=True)

            use_fuzzy = st.checkbox("Enable fuzzy matching for area names", value=False)

            first_areas = summary_df['Area Name / Block'].dropna().astype(str).tolist()
            second_areas = df2[area_col].astype(str).tolist()

            matched_areas = []
            for area in second_areas:
                matched = match_area_name(area, first_areas, use_fuzzy=use_fuzzy)
                matched_areas.append(matched if matched else "No Match")

            df2['Matched Area'] = matched_areas

            st.subheader("🔍 Matching Results (Second file areas matched to first file areas)")
            st.dataframe(df2[[area_col, 'Matched Area', media_col]], use_container_width=True)

            st.subheader("➕ Adjust counts from second file (Add second sheet's 'No. of Media' to combined first sheets counts)")

            adjusted_counts = summary_df.copy()
            adjusted_counts['Entry in SAPW'] = "Yes"
            adjusted_counts['Entry in Excel'] = "Yes"

            counts_dict = dict(zip(adjusted_counts['Area Name / Block'], adjusted_counts['No. of Media']))

            for idx, row in df2.iterrows():
                matched_area = row['Matched Area']
                second_media = row[media_col]
                if matched_area != "No Match" and pd.notna(second_media):
                    try:
                        second_media_val = float(second_media)
                        counts_dict[matched_area] = counts_dict.get(matched_area, 0) + second_media_val
                    except ValueError:
                        st.warning(f"Invalid 'No. of Media' at row {idx + 3}: {second_media}")

            adjusted_counts['No. of Media'] = adjusted_counts['Area Name / Block'].map(counts_dict).fillna(0).astype(int)

            adjusted_counts = adjusted_counts.reset_index(drop=True)
            adjusted_counts.index += 1
            adjusted_counts = adjusted_counts.rename_axis('S.No').reset_index()

            adjusted_counts = adjusted_counts[['S.No', 'Area Name / Block', 'No. of Media', 'Entry in SAPW', 'Entry in Excel']]

            st.subheader("✅ Adjusted Area Media Counts")
            st.dataframe(adjusted_counts, use_container_width=True)

            buffer = BytesIO()
            adjusted_counts.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                label="📥 Download Adjusted Summary as Excel",
                data=buffer,
                file_name="adjusted_area_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please upload one or more Excel files to begin analysis.")

if __name__ == "__main__":
    analyze_app()
