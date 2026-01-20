import streamlit as st
import os
import glob
import pandas as pd
from paper_scraper_oop import PaperScraper, DataManager, Visualizer, PaperAnalyzer

st.set_page_config(page_title="CV Paper Scraper & Analyzer", layout="wide")

st.title("CV Paper Scraper & Analyzer")
st.markdown("""
This tool allows you to scrape papers from top computer vision conferences (CVPR, ICCV, NeurIPS, ECCV) 
and visualize the trends, categories, and top authors.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")
conference = st.sidebar.selectbox(
    "Conference",
    ["cvpr", "iccv", "neurips", "eccv"],
    index=0
)
year_options = {
    "cvpr": ["2025", "2024", "2023", "2022"],
    "iccv": ["2025", "2023", "2021", "2019"],
    "eccv": ["2024", "2022", "2020", "2018"],
    "neurips": ["2025", "2024", "2023", "2022"],
}
year = st.sidebar.selectbox("Year", year_options.get(conference, []))
max_workers = st.sidebar.slider("Threads (Workers)", min_value=1, max_value=50, value=20)

# Main Content - Tabs
tab1, tab2 = st.tabs(["Scraper", "Analysis & Visualization"])

# --- Tab 1: Scraper ---
with tab1:
    st.header(f"Scrape {conference.upper()} {year}")
    
    st.info("Click the button below to start scraping. This may take a while depending on the number of papers.")
    
    if st.button("Start Scraping", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            with st.spinner(f"Gathering treasures from {conference.upper()} {year}... Feel free to do anything you want except close the browser (◍•ᴗ•◍)ﾉ♡"):
                # Initialize Scraper
                scraper = PaperScraper(conference, year)
                def update_progress(done, total, errors):
                    if total > 0:
                        progress_bar.progress(done / total)
                    else:
                        progress_bar.progress(0)
                    status_text.text(f"Progress: {done}/{total} | Errors: {errors}")
                papers = scraper.scrape(max_workers=max_workers, progress_callback=update_progress)
                
                if papers:
                    st.success(f"Successfully scraped {len(papers)} papers!")
                    
                    # Save Data
                    csv_file = DataManager.save(papers, conference, year)
                    st.info(f"Data saved to `{csv_file}`")
                    
                    # Show Preview
                    st.subheader("Data Preview")
                    df = pd.DataFrame(papers)
                    st.dataframe(df.head(10))
                    
                    # Download Button
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name=csv_file,
                            mime="text/csv"
                        )
                else:
                    st.error("No papers found. Please check the conference/year or your internet connection.")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Tab 2: Analysis ---
with tab2:
    st.header("Analysis & Visualization")
    
    # Check for all available CSV files (recursively)
    csv_files = glob.glob("**/*_papers.csv", recursive=True)
    
    if not csv_files:
        st.warning("No data files found. Please scrape some data first in the 'Scraper' tab.")
    else:
        # Sort files to have recent ones (or organized ones) first or alphabetically
        csv_files.sort()
        
        # Default to current selection if it exists, otherwise first file
        # We need to reconstruct the expected path for the current selection to match
        current_csv_name = f'{conference}_{year}_papers.csv'
        # Try to find a match in the list that ends with this name
        default_index = 0
        for i, f in enumerate(csv_files):
            if f.endswith(current_csv_name):
                default_index = i
                break
            
        selected_file = st.selectbox(
            "Select Dataset to Analyze", 
            csv_files, 
            index=default_index,
            help="Select from locally saved paper datasets."
        )
        
        # Parse conference and year from filename
        try:
            # Filename format: {conference}_{year}_papers.csv
            # Handle paths (e.g. CVPR_2024/cvpr_2024_papers.csv)
            basename = os.path.basename(selected_file)
            parts = basename.split('_')
            sel_conf = parts[0]
            sel_year = parts[1]
            
            st.info(f"Selected Data: **{sel_conf.upper()} {sel_year}**")
            
            # Use session state to persist data across reruns (e.g. when filtering)
            if 'analysis_data' not in st.session_state:
                st.session_state.analysis_data = None
            if 'analysis_conf_year' not in st.session_state:
                st.session_state.analysis_conf_year = None
            if 'search_term' not in st.session_state:
                st.session_state.search_term = ""
            if 'cat_filter' not in st.session_state:
                st.session_state.cat_filter = "All"
            
            if st.button("Run Analysis & Generate Plots", type = "primary"):
                try:
                    with st.spinner(f"Analyzing {sel_conf.upper()} {sel_year} data... You know what I want to say...Feel free to do anything you want except close the browser (◍•ᴗ•◍)ﾉ♡"):
                        # Load Data
                        papers = DataManager.load(sel_conf, sel_year, file_path=selected_file)
                        
                        if not papers:
                            st.error("Loaded data is empty.")
                        else:
                            # Save to session state
                            st.session_state.analysis_data = papers
                            st.session_state.analysis_conf_year = (sel_conf, sel_year)
                            # Reset filters
                            st.session_state.search_term = ""
                            st.session_state.cat_filter = "All"
                            st.success("Analysis Complete!")
                            
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
            
            # Display Analysis if data is loaded
            if st.session_state.analysis_data:
                papers = st.session_state.analysis_data
                conf, yr = st.session_state.analysis_conf_year
                
                # Check if the loaded data matches current selection (optional warning)
                if conf != sel_conf or yr != sel_year:
                    st.warning(f"Displaying analysis for {conf.upper()} {yr}. Click 'Run Analysis' to update for {sel_conf.upper()} {sel_year}.")

                # 1. Categories
                st.subheader("1. Paper Categories")
                fig_cat = Visualizer.plot_categories_interactive(papers, conf, yr)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    cat_img = Visualizer.plot_categories(papers, conf, yr)
                    if cat_img and os.path.exists(cat_img):
                        st.image(cat_img, caption="Category Distribution")
                    else:
                        st.warning("Category plot not generated.")
                
                # 2. Word Cloud
                st.subheader("2. Word Cloud")
                # Wordcloud is static image, regenerate only if needed or just show cached image if exists
                # For simplicity, we call generate again which usually saves to file. 
                # Optimization: Check if file exists first to avoid re-generating on every interaction?
                # But generate_wordcloud is fast enough or we accept it for now. 
                # Let's try to load existing image first.
                folder = f"{conf}_{yr}"
                wc_filename = f'{conf}_{yr}_wordcloud.png'
                wc_path = os.path.join(folder, wc_filename)
                
                if os.path.exists(wc_path):
                     st.image(wc_path, caption="Word Cloud")
                else:
                     wc_img = Visualizer.generate_wordcloud(papers, conf, yr)
                     if wc_img and os.path.exists(wc_img):
                        st.image(wc_img, caption="Word Cloud")
                     else:
                        st.warning("Word cloud not generated.")

                # 3. Top Authors
                st.subheader("3. Top Authors")
                fig_auth = Visualizer.plot_top_authors_interactive(papers, conf, yr)
                if fig_auth:
                    st.plotly_chart(fig_auth, use_container_width=True)
                else:
                    auth_img = Visualizer.plot_top_authors(papers, conf, yr)
                    if auth_img and os.path.exists(auth_img):
                        st.image(auth_img, caption="Top Authors")
                    else:
                        st.warning("Top authors plot not generated.")
                
                st.divider()
                st.header("4. Data Explorer")
                
                # Convert to DataFrame
                df_papers = pd.DataFrame(papers)
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Search (Title/Abstract)", key="search_term")
                with col2:
                    all_cats = ["All"] + sorted(list(set(df_papers['category'])))
                    # Ensure current filter is valid
                    if st.session_state.cat_filter not in all_cats:
                        st.session_state.cat_filter = "All"
                    st.selectbox("Filter by Category", all_cats, key="cat_filter")
                    
                # Apply filters
                filtered_df = df_papers.copy()
                if st.session_state.cat_filter != "All":
                    filtered_df = filtered_df[filtered_df['category'] == st.session_state.cat_filter]
                    
                if st.session_state.search_term:
                    filtered_df = filtered_df[
                        filtered_df['title'].str.contains(st.session_state.search_term, case=False, na=False) | 
                        filtered_df['abstract'].str.contains(st.session_state.search_term, case=False, na=False)
                    ]
                    
                st.write(f"Showing {len(filtered_df)} papers")
                st.dataframe(filtered_df)
                                
        except IndexError:
             st.error("Invalid filename format. Expected '{conference}_{year}_papers.csv'.")
