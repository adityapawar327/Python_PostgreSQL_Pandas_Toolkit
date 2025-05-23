import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from io import BytesIO
from fuzzywuzzy import process

def analyze_media_id_ranges(df, media_col='Media_ID', area_col='Area Name / Block'):
    """
    Analyze Media_ID consecutive ranges by area and identify jumps.
    Count unique values only for consecutive IDs.
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Ensure columns exist
    if media_col not in df_copy.columns or area_col not in df_copy.columns:
        st.error(f"Required columns '{media_col}' or '{area_col}' not found in the data.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Drop rows with missing values in required columns
    df_copy = df_copy.dropna(subset=[media_col, area_col])
    
    # Extract area name and media ID
    areas = df_copy[area_col].unique()
    
    # Prepare results dataframes
    ranges_data = []  # For Table 1: Media Id Ranges
    summary_data = []  # For Table 2: Summary by Area
    
    # Process each area
    for area in areas:
        area_df = df_copy[df_copy[area_col] == area]
        # Get unique media IDs in this area
        unique_media_ids = area_df[media_col].astype(str).unique().tolist()
        
        # Parse Media IDs to extract prefix, number and suffix parts
        parsed_ids = []
        for media_id in unique_media_ids:
            match = re.match(r'(T\d+)(\d{2})([A-Z]+)', media_id)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                suffix = match.group(3)
                parsed_ids.append({
                    'full_id': media_id,
                    'prefix': prefix,
                    'number': number,
                    'suffix': suffix,
                    'base': f"{prefix}{suffix}"
                })
        
        # Group by base (prefix+suffix)
        grouped_ids = defaultdict(list)
        for parsed in parsed_ids:
            grouped_ids[parsed['base']].append(parsed)
        
        # Process each group to find consecutive sequences
        area_ranges = []
        for base, ids in grouped_ids.items():
            # Sort by number
            sorted_ids = sorted(ids, key=lambda x: x['number'])
            
            # Find consecutive sequences
            if sorted_ids:
                start_idx = 0
                for i in range(1, len(sorted_ids)):
                    # Check if there's a jump (not consecutive)
                    if sorted_ids[i]['number'] != sorted_ids[i-1]['number'] + 1:
                        # Add the completed range
                        start_id = sorted_ids[start_idx]['full_id']
                        end_id = sorted_ids[i-1]['full_id']
                        count = i - start_idx
                        jump_note = "Jump here" if i < len(sorted_ids) else ""
                        
                        area_ranges.append({
                            'area': area,
                            'start_id': start_id,
                            'end_id': end_id,
                            'count': count,
                            'note': jump_note
                        })
                        
                        # Start a new range
                        start_idx = i
                
                # Add the last range
                start_id = sorted_ids[start_idx]['full_id']
                end_id = sorted_ids[-1]['full_id']
                count = len(sorted_ids) - start_idx
                
                area_ranges.append({
                    'area': area,
                    'start_id': start_id,
                    'end_id': end_id,
                    'count': count,
                    'note': ""
                })
        
        # Add all ranges for this area to the results
        ranges_data.extend(area_ranges)
        
        # Add to summary - count unique media IDs only
        summary_data.append({
            'area': area,
            'count': len(unique_media_ids)
        })
    
    # Create DataFrames for the two tables
    ranges_df = pd.DataFrame(ranges_data)
    summary_df = pd.DataFrame(summary_data)
    
    # Format ranges dataframe
    if not ranges_df.empty:
        ranges_df = ranges_df.rename(columns={
            'area': 'Area Name / Block',
            'start_id': 'start',
            'end_id': 'end',
            'count': 'Count',
            'note': 'Note'
        })
        ranges_df['Media Id Range'] = ranges_df.apply(
            lambda row: f"{row['start']} - {row['end']}", axis=1
        )
        ranges_df = ranges_df[['Area Name / Block', 'Media Id Range', 'Count', 'Note']]
        ranges_df = ranges_df.reset_index(drop=True)
        ranges_df.index += 1  # Start from 1 instead of 0
        ranges_df = ranges_df.rename_axis('Sr. No').reset_index()
    
    # Format summary dataframe
    if not summary_df.empty:
        summary_df = summary_df.rename(columns={
            'area': 'Area Name / Block',
            'count': 'No. of Media Ids'
        })
        summary_df = summary_df.reset_index(drop=True)
        summary_df.index += 1  # Start from 1 instead of 0
        summary_df = summary_df.rename_axis('Sr. No').reset_index()
        
        # Add Total row
        total_count = summary_df['No. of Media Ids'].sum()
        total_row = pd.DataFrame({
            'Sr. No': ['Total'],
            'Area Name / Block': ['Total'],
            'No. of Media Ids': [total_count]
        })
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)
    
    return ranges_df, summary_df

def summarize_media_by_area(df, media_col='Media_ID', area_col='Area Name / Block'):
    """
    Original function to summarize media counts by area.
    This function now counts unique Media IDs per area.
    """
    df = df.dropna(subset=[media_col, area_col])
    summary = df.groupby(area_col)[media_col].nunique().reset_index()
    summary.rename(columns={media_col: 'No. of Media'}, inplace=True)
    summary['No. of Media'] = summary['No. of Media'].astype(int)
    summary.reset_index(drop=True, inplace=True)
    return summary

def match_area_name(area_name, area_list, threshold=80, use_fuzzy=False):
    """
    Match area name with potential matches from a list.
    """
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
    st.title("🔍 Analyze Media_ID Ranges and Area Summary")

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

        # Generate the two tables
        ranges_df, summary_df = analyze_media_id_ranges(combined_df)
        
        # Display Table 1: Media Id Ranges
        st.subheader("Table 1: Media Id Ranges")
        st.dataframe(ranges_df, use_container_width=True)
        
        # Display Table 2: Summary by Area
        st.subheader("Table 2: Summary by Area Name / Block")
        st.dataframe(summary_df, use_container_width=True)
        
        # Option to download the tables
        buffer1 = BytesIO()
        ranges_df.to_excel(buffer1, index=False, engine='openpyxl')
        buffer1.seek(0)
        st.download_button(
            label="📥 Download Media Id Ranges as Excel",
            data=buffer1,
            file_name="media_id_ranges.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        buffer2 = BytesIO()
        summary_df.to_excel(buffer2, index=False, engine='openpyxl')
        buffer2.seek(0)
        st.download_button(
            label="📥 Download Area Summary as Excel",
            data=buffer2,
            file_name="area_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Optional: Add second file processing functionality if needed
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
            # Remove "Total" row from first_areas if present
            if 'Total' in first_areas:
                first_areas.remove('Total')
                
            second_areas = df2[area_col].astype(str).tolist()

            matched_areas = []
            for area in second_areas:
                matched = match_area_name(area, first_areas, use_fuzzy=use_fuzzy)
                matched_areas.append(matched if matched else "No Match")

            df2['Matched Area'] = matched_areas

            st.subheader("🔍 Matching Results (Second file areas matched to first file areas)")
            st.dataframe(df2[[area_col, 'Matched Area', media_col]], use_container_width=True)

            st.subheader("➕ Adjust counts from second file (Add second sheet's 'No. of Media' to combined first sheets counts)")

            # Get only non-Total rows from summary_df
            adjusted_counts = summary_df[summary_df['Sr. No'] != 'Total'].copy()
            adjusted_counts['Entry in SAPW'] = "Yes"
            adjusted_counts['Entry in Excel'] = "Yes"

            counts_dict = dict(zip(adjusted_counts['Area Name / Block'], adjusted_counts['No. of Media Ids']))

            for idx, row in df2.iterrows():
                matched_area = row['Matched Area']
                second_media = row[media_col]
                if matched_area != "No Match" and pd.notna(second_media):
                    try:
                        second_media_val = float(second_media)
                        counts_dict[matched_area] = counts_dict.get(matched_area, 0) + second_media_val
                    except ValueError:
                        st.warning(f"Invalid 'No. of Media' at row {idx + 3}: {second_media}")

            adjusted_counts['No. of Media Ids'] = adjusted_counts['Area Name / Block'].map(counts_dict).fillna(0).astype(int)

            # Re-index adjusted_counts
            adjusted_counts = adjusted_counts.reset_index(drop=True)
            adjusted_counts['Sr. No'] = adjusted_counts.index + 1
            
            # Create total row for adjusted counts
            total_adjusted = adjusted_counts['No. of Media Ids'].sum()
            total_row = pd.DataFrame({
                'Sr. No': ['Total'],
                'Area Name / Block': ['Total'],
                'No. of Media Ids': [total_adjusted],
                'Entry in SAPW': [""],
                'Entry in Excel': [""]
            })
            
            adjusted_counts = pd.concat([adjusted_counts, total_row], ignore_index=True)
            adjusted_counts = adjusted_counts[['Sr. No', 'Area Name / Block', 'No. of Media Ids', 'Entry in SAPW', 'Entry in Excel']]

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