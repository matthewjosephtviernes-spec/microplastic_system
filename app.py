import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# Page configuration
st.set_page_config(
    page_title="Streamlit Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data storage
if 'items' not in st.session_state:
    st.session_state.items = []
if 'counter' not in st.session_state:
    st.session_state.counter = 1

# Title and description
st.markdown('<h1 class="main-header">🚀 Streamlit Dashboard App</h1>', unsafe_allow_html=True)
st.markdown("A modern web app built with Streamlit")

# Sidebar
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title("Navigation")
    
    menu = st.radio(
        "Select Page",
        ["🏠 Dashboard", "📊 Add Items", "📈 Analytics", "⚙️ Settings"]
    )
    
    st.divider()
    
    # Sidebar metrics
    st.subheader("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Items", len(st.session_state.items))
    with col2:
        st.metric("App Visits", st.session_state.counter)
    
    st.divider()
    
    # Export data
    if st.button("📥 Export Data as JSON"):
        if st.session_state.items:
            json_data = json.dumps(st.session_state.items, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No data to export")

# Dashboard Page
if menu == "🏠 Dashboard":
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Items", len(st.session_state.items), delta="+0" if not st.session_state.items else f"+{len(st.session_state.items)}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active", len(st.session_state.items), "items")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Current Time", current_time)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Status", "Online", delta="Running")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Items table
    st.subheader("📋 Items List")
    
    if st.session_state.items:
        # Convert items to DataFrame for display
        df = pd.DataFrame(st.session_state.items)
        
        # Display with tabs
        tab1, tab2 = st.tabs(["📊 Table View", "📈 Chart View"])
        
        with tab1:
            # Add a search bar
            search_term = st.text_input("🔍 Search items...", placeholder="Type to filter items")
            
            if search_term:
                filtered_df = df[df['name'].str.contains(search_term, case=False, na=False) | 
                                 df['description'].str.contains(search_term, case=False, na=False)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
            
            # Delete functionality
            st.subheader("Delete Items")
            items_to_delete = st.multiselect(
                "Select items to delete:",
                options=df['name'].tolist(),
                format_func=lambda x: f"{x} (ID: {df[df['name']==x]['id'].values[0]})"
            )
            
            if items_to_delete and st.button("🗑️ Delete Selected", type="secondary"):
                st.session_state.items = [item for item in st.session_state.items 
                                         if item['name'] not in items_to_delete]
                st.rerun()
        
        with tab2:
            if len(df) > 0:
                # Create visualizations
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Bar chart of items by creation date
                    df['date'] = pd.to_datetime(df['created_at']).dt.date
                    items_by_date = df.groupby('date').size().reset_index(name='count')
                    
                    fig1 = px.bar(
                        items_by_date,
                        x='date',
                        y='count',
                        title="Items Added Over Time",
                        color='count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_col2:
                    # Pie chart of item categories (if available)
                    fig2 = px.pie(
                        df,
                        names='category' if 'category' in df.columns else 'name',
                        title="Items Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("No items yet. Go to 'Add Items' page to add your first item!")
        
        # Quick add form on dashboard
        with st.expander("➕ Quick Add Item"):
            with st.form("quick_add_form"):
                quick_name = st.text_input("Item Name")
                quick_desc = st.text_area("Description")
                quick_submit = st.form_submit_button("Add Item")
                
                if quick_submit and quick_name:
                    new_item = {
                        'id': st.session_state.counter,
                        'name': quick_name,
                        'description': quick_desc,
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'category': 'Quick Add'
                    }
                    st.session_state.items.append(new_item)
                    st.session_state.counter += 1
                    st.success(f"Item '{quick_name}' added!")
                    time.sleep(0.5)
                    st.rerun()

# Add Items Page
elif menu == "📊 Add Items":
    st.header("➕ Add New Items")
    
    # Two column layout for form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("add_item_form"):
            name = st.text_input("Item Name *", placeholder="Enter item name")
            description = st.text_area("Description", placeholder="Enter item description", height=100)
            category = st.selectbox("Category", ["General", "Electronics", "Books", "Clothing", "Other"])
            priority = st.slider("Priority", 1, 5, 3)
            
            # File upload
            uploaded_file = st.file_uploader("Upload related file", type=['txt', 'pdf', 'png', 'jpg'])
            
            col_a, col_b = st.columns(2)
            with col_a:
                submitted = st.form_submit_button("💾 Save Item", type="primary")
            with col_b:
                clear_form = st.form_submit_button("🧹 Clear Form", type="secondary")
            
            if submitted and name:
                new_item = {
                    'id': st.session_state.counter,
                    'name': name,
                    'description': description,
                    'category': category,
                    'priority': priority,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'has_file': uploaded_file is not None
                }
                
                st.session_state.items.append(new_item)
                st.session_state.counter += 1
                
                if uploaded_file:
                    # Save uploaded file info
                    st.success(f"Item '{name}' with file uploaded successfully!")
                else:
                    st.success(f"Item '{name}' added successfully!")
                
                time.sleep(1)
                st.rerun()
            
            if clear_form:
                st.rerun()
    
    with col2:
        st.info("💡 Tips:")
        st.markdown("""
        - * marks required fields
        - Use descriptive names
        - Categorize items properly
        - Upload relevant files
        """)
        
        # Preview
        st.subheader("Preview")
        if st.session_state.items:
            latest_item = st.session_state.items[-1]
            st.json(latest_item)
        else:
            st.write("No items yet")

# Analytics Page
elif menu == "📈 Analytics":
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.items:
        df = pd.DataFrame(st.session_state.items)
        
        # Create metrics row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Items", len(df))
        with mcol2:
            st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 1)
        with mcol3:
            avg_priority = df['priority'].mean() if 'priority' in df.columns else 0
            st.metric("Avg Priority", f"{avg_priority:.1f}")
        with mcol4:
            latest_date = pd.to_datetime(df['created_at']).max().strftime("%b %d")
            st.metric("Latest Add", latest_date)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                fig1 = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Items by Category",
                    hole=0.3
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Timeline chart
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            timeline_df = df.groupby('date').size().cumsum().reset_index(name='cumulative')
            
            fig2 = px.line(
                timeline_df,
                x='date',
                y='cumulative',
                title="Cumulative Items Over Time",
                markers=True
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Priority heatmap
        st.subheader("Priority Analysis")
        if 'priority' in df.columns and 'category' in df.columns:
            priority_matrix = pd.crosstab(df['category'], df['priority'])
            fig3 = px.imshow(
                priority_matrix,
                title="Priority Heatmap by Category",
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("No data available for analytics. Add some items first!")

# Settings Page
elif menu == "⚙️ Settings":
    st.header("Settings")
    
    tab1, tab2, tab3 = st.tabs(["App Settings", "Data Management", "About"])
    
    with tab1:
        st.subheader("App Configuration")
        
        # Theme settings
        theme = st.selectbox("Theme", ["Light", "Dark", "System"])
        st.checkbox("Enable notifications", value=True)
        st.checkbox("Show previews", value=True)
        
        # Performance settings
        st.subheader("Performance")
        cache_size = st.slider("Cache size (MB)", 10, 1000, 100)
        refresh_rate = st.select_slider("Auto-refresh rate", options=["Off", "30s", "1m", "5m", "10m"])
        
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved!")
    
    with tab2:
        st.subheader("Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Backup Data**")
            if st.button("Create Backup"):
                backup_data = {
                    'items': st.session_state.items,
                    'counter': st.session_state.counter,
                    'backup_time': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download Backup",
                    data=json.dumps(backup_data, indent=2),
                    file_name=f"backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.write("**Restore Data**")
            uploaded_backup = st.file_uploader("Choose backup file", type=['json'])
            if uploaded_backup and st.button("Restore from Backup"):
                try:
                    backup = json.load(uploaded_backup)
                    st.session_state.items = backup.get('items', [])
                    st.session_state.counter = backup.get('counter', 1)
                    st.success("Data restored successfully!")
                    time.sleep(1)
                    st.rerun()
                except:
                    st.error("Invalid backup file")
        
        st.divider()
        
        st.write("**Danger Zone**")
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                if st.button("Confirm Delete", type="primary"):
                    st.session_state.items = []
                    st.session_state.counter = 1
                    st.warning("All data has been deleted!")
                    time.sleep(1)
                    st.rerun()
    
    with tab3:
        st.subheader("About This App")
        st.markdown("""
        ### Streamlit Dashboard App
        
        **Version:** 1.0.0
        
        **Description:**
        A modern web application built with Streamlit for managing items and visualizing data.
        
        **Features:**
        - 📋 Item management with CRUD operations
        - 📊 Interactive data visualizations
        - 📈 Real-time analytics
        - ⚙️ Configurable settings
        - 📥 Data import/export
        
        **Built with:**
        - Streamlit
        - Plotly
        - Pandas
        
        **GitHub:** [Your Repository Link]
        """)
        
        st.divider()
        
        # System info
        st.write("**System Information:**")
        st.code(f"""
        App Status: Running
        Items in memory: {len(st.session_state.items)}
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Python version: 3.x
        """)

# Footer
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption(f"© {datetime.now().year} Streamlit App")
with footer_col2:
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
with footer_col3:
    if st.button("🔄 Refresh App"):
        st.rerun()
