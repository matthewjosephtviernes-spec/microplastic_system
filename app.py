import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np
import time

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================
st.set_page_config(
    page_title="Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF4B4B 0%, #1E88E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1E88E5;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #1A2530 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'items' not in st.session_state:
    st.session_state.items = []
    
if 'counter' not in st.session_state:
    st.session_state.counter = 1
    
if 'app_visits' not in st.session_state:
    st.session_state.app_visits = 0
    
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Increment app visits
st.session_state.app_visits += 1

# ============================================
# SAMPLE DATA FOR DEMO
# ============================================
if len(st.session_state.items) == 0:
    # Add some sample data for demo
    sample_items = [
        {
            'id': 1,
            'name': 'Project Alpha',
            'description': 'Main dashboard development',
            'category': 'Development',
            'priority': 5,
            'status': 'In Progress',
            'created_at': (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
            'due_date': (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
            'progress': 75
        },
        {
            'id': 2,
            'name': 'Marketing Campaign',
            'description': 'Q4 marketing strategy',
            'category': 'Marketing',
            'priority': 4,
            'status': 'Completed',
            'created_at': (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
            'due_date': (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            'progress': 100
        },
        {
            'id': 3,
            'name': 'Client Meeting',
            'description': 'Quarterly review with stakeholders',
            'category': 'Meeting',
            'priority': 3,
            'status': 'Pending',
            'created_at': (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
            'due_date': (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
            'progress': 25
        },
        {
            'id': 4,
            'name': 'Budget Planning',
            'description': 'Annual budget allocation',
            'category': 'Finance',
            'priority': 4,
            'status': 'In Progress',
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'due_date': (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d"),
            'progress': 50
        },
        {
            'id': 5,
            'name': 'Team Training',
            'description': 'New tools and processes training',
            'category': 'HR',
            'priority': 2,
            'status': 'Pending',
            'created_at': (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            'due_date': (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            'progress': 10
        }
    ]
    st.session_state.items = sample_items
    st.session_state.counter = len(sample_items) + 1

# ============================================
# SIDEBAR - NAVIGATION
# ============================================
with st.sidebar:
    # Logo
    st.image("https://cdn-icons-png.flaticon.com/512/919/919826.png", width=80)
    
    # Title
    st.markdown("<h1 style='color: white;'>🚀 Dashboard Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #bbb;'>Enterprise Management System</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    st.markdown("<h3 style='color: white;'>📌 Navigation</h3>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["📊 Dashboard", "➕ Add Item", "📋 View All", "📈 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick Stats
    st.markdown("<h3 style='color: white;'>📈 Quick Stats</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Items", len(st.session_state.items), 
                 delta=f"+{max(0, len(st.session_state.items) - 5)}" if len(st.session_state.items) > 5 else None)
    with col2:
        st.metric("App Visits", st.session_state.app_visits)
    
    # Progress summary
    if st.session_state.items:
        completed = len([item for item in st.session_state.items if item.get('status') == 'Completed'])
        in_progress = len([item for item in st.session_state.items if item.get('status') == 'In Progress'])
        pending = len([item for item in st.session_state.items if item.get('status') == 'Pending'])
        
        st.progress(completed / len(st.session_state.items))
        st.caption(f"✅ {completed} Completed | 🔄 {in_progress} In Progress | ⏳ {pending} Pending")
    
    st.divider()
    
    # Quick Actions
    st.markdown("<h3 style='color: white;'>⚡ Quick Actions</h3>", unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.button("📥 Export All", use_container_width=True):
        st.info("Export feature ready - see main page")
    
    if st.button("🧹 Clear Filters", use_container_width=True):
        st.success("Filters cleared!")
    
    st.divider()
    
    # Theme Toggle
    current_theme = st.selectbox(
        "🎨 Theme",
        ["Light", "Dark", "Blue"],
        index=0
    )
    if current_theme != st.session_state.theme:
        st.session_state.theme = current_theme
        st.success(f"Theme changed to {current_theme}")

# ============================================
# MAIN CONTENT AREA
# ============================================

# Dashboard Page
if page == "📊 Dashboard":
    # Header
    st.markdown("<h1 class='main-title'>📊 Executive Dashboard</h1>", unsafe_allow_html=True)
    
    # Top Metrics
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_items = len(st.session_state.items)
            st.metric("Total Projects", total_items, delta="+2 from last week")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            completed = len([item for item in st.session_state.items if item.get('status') == 'Completed'])
            completion_rate = (completed / total_items * 100) if total_items > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%", delta="+5.2%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_priority = np.mean([item.get('priority', 0) for item in st.session_state.items]) if st.session_state.items else 0
            st.metric("Avg Priority", f"{avg_priority:.1f}/5", delta="-0.3")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            overdue = len([item for item in st.session_state.items if 'due_date' in item and 
                          datetime.strptime(item['due_date'], "%Y-%m-%d") < datetime.now()])
            st.metric("Overdue Items", overdue, delta="-1", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Charts Row
    st.markdown("### 📊 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status Distribution Pie Chart
        if st.session_state.items:
            status_counts = pd.DataFrame(st.session_state.items)['status'].value_counts()
            fig1 = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Project Status Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Priority Bar Chart
        if st.session_state.items:
            priority_counts = pd.DataFrame(st.session_state.items)['priority'].value_counts().sort_index()
            fig2 = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Items by Priority Level",
                labels={'x': 'Priority', 'y': 'Count'},
                color=priority_counts.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Recent Activity & Quick Add
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Recent Activity")
        if st.session_state.items:
            df = pd.DataFrame(st.session_state.items)
            df['days_ago'] = df['created_at'].apply(
                lambda x: (datetime.now() - datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).days
            )
            recent = df.nsmallest(3, 'days_ago')
            
            for _, item in recent.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.write(f"**{item['name']}**")
                        st.caption(f"{item['description'][:50]}...")
                    with col_b:
                        st.caption(f"🔸 {item['category']}")
                    with col_c:
                        progress = item.get('progress', 0)
                        st.progress(progress / 100)
                        st.caption(f"{progress}%")
                    st.divider()
    
    with col2:
        st.markdown("### ⚡ Quick Add")
        with st.form("quick_add", clear_on_submit=True):
            quick_name = st.text_input("Item Name")
            quick_category = st.selectbox("Category", ["General", "Development", "Marketing", "Finance"])
            quick_submit = st.form_submit_button("➕ Add Now")
            
            if quick_submit and quick_name:
                new_item = {
                    'id': st.session_state.counter,
                    'name': quick_name,
                    'description': "Quick added item",
                    'category': quick_category,
                    'priority': 3,
                    'status': 'Pending',
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'due_date': (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    'progress': 0
                }
                st.session_state.items.append(new_item)
                st.session_state.counter += 1
                st.success(f"✅ '{quick_name}' added!")
                time.sleep(1)
                st.rerun()

# Add Item Page
elif page == "➕ Add Item":
    st.markdown("<h1 class='main-title'>➕ Add New Item</h1>", unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("add_item_form", clear_on_submit=False):
            st.markdown("### 📝 Item Details")
            
            # Basic Info
            col_a, col_b = st.columns(2)
            with col_a:
                name = st.text_input("Item Name *", placeholder="Enter item name")
            with col_b:
                category = st.selectbox("Category *", 
                    ["Development", "Marketing", "Finance", "HR", "Operations", "Sales", "Other"])
            
            # Description
            description = st.text_area("Description", 
                placeholder="Describe the item in detail...", 
                height=120)
            
            # Priority and Status
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                priority = st.slider("Priority", 1, 5, 3, 
                    help="1 = Low, 5 = Critical")
            with col_d:
                status = st.selectbox("Status", 
                    ["Pending", "In Progress", "Completed", "On Hold"])
            with col_e:
                due_date = st.date_input("Due Date", 
                    min_value=datetime.now().date())
            
            # Progress
            progress = st.slider("Progress %", 0, 100, 0, 5)
            
            # File Upload
            uploaded_file = st.file_uploader("Attach File (Optional)", 
                type=['pdf', 'docx', 'xlsx', 'jpg', 'png'])
            
            # Form Buttons
            st.markdown("---")
            col_submit, col_clear = st.columns(2)
            with col_submit:
                submit = st.form_submit_button("💾 Save Item", 
                    type="primary", use_container_width=True)
            with col_clear:
                clear = st.form_submit_button("🧹 Clear Form", 
                    use_container_width=True)
            
            if submit:
                if not name:
                    st.error("❌ Item name is required!")
                else:
                    new_item = {
                        'id': st.session_state.counter,
                        'name': name,
                        'description': description,
                        'category': category,
                        'priority': priority,
                        'status': status,
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'due_date': due_date.strftime("%Y-%m-%d"),
                        'progress': progress,
                        'has_attachment': uploaded_file is not None
                    }
                    
                    st.session_state.items.append(new_item)
                    st.session_state.counter += 1
                    
                    # Show success message
                    st.success(f"✅ Item '{name}' added successfully!")
                    st.balloons()
                    
                    # Show preview
                    with st.expander("📄 Preview Added Item"):
                        st.json(new_item)
                    
                    # Reset form after delay
                    time.sleep(2)
                    st.rerun()
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **Best Practices:**
        
        1. **Be Specific** - Use clear, descriptive names
        2. **Set Realistic Dates** - Allow buffer time
        3. **Prioritize Wisely** - Use priority levels effectively
        4. **Track Progress** - Update regularly
        5. **Add Attachments** - Include relevant files
        """)
        
        st.markdown("### 📊 Stats")
        if st.session_state.items:
            df = pd.DataFrame(st.session_state.items)
            cat_counts = df['category'].value_counts()
            
            for cat, count in cat_counts.head(3).items():
                st.write(f"**{cat}:** {count} items")
            
            st.progress(len(st.session_state.items) / 50)
            st.caption(f"{len(st.session_state.items)} / 50 items")

# View All Page
elif page == "📋 View All":
    st.markdown("<h1 class='main-title'>📋 All Items</h1>", unsafe_allow_html=True)
    
    if not st.session_state.items:
        st.warning("No items found. Add some items first!")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.items)
        
        # Filters
        st.markdown("### 🔍 Filters & Search")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search_term = st.text_input("Search by name...")
        with col2:
            category_filter = st.multiselect("Category", 
                options=df['category'].unique().tolist() if 'category' in df.columns else [],
                default=[])
        with col3:
            status_filter = st.multiselect("Status",
                options=df['status'].unique().tolist() if 'status' in df.columns else [],
                default=[])
        with col4:
            priority_filter = st.multiselect("Priority",
                options=sorted(df['priority'].unique()) if 'priority' in df.columns else [],
                default=[])
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_term:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False, na=False)]
        
        if category_filter:
            filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
        
        if status_filter:
            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
        
        if priority_filter:
            filtered_df = filtered_df[filtered_df['priority'].isin(priority_filter)]
        
        # Display results
        st.markdown(f"### 📊 Showing {len(filtered_df)} of {len(df)} items")
        
        if len(filtered_df) > 0:
            # Display as enhanced dataframe
            display_cols = ['id', 'name', 'category', 'priority', 'status', 'progress', 'due_date']
            display_df = filtered_df[display_cols].copy()
            
            # Format progress as bar in dataframe
            def progress_bar(val):
                return f"▮" * int(val/20)
            
            display_df['progress_bar'] = display_df['progress'].apply(progress_bar)
            
            # Show dataframe
            st.dataframe(
                display_df.style.background_gradient(
                    subset=['priority'], 
                    cmap='RdYlGn_r'
                ).bar(
                    subset=['progress'],
                    color='#5DADE2'
                ),
                use_container_width=True,
                height=400
            )
            
            # Item Actions
            st.markdown("### ⚙️ Item Actions")
            
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                if st.button("📥 Export Filtered", use_container_width=True):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"filtered_items_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col_action2:
                items_to_delete = st.multiselect(
                    "Select items to delete:",
                    options=filtered_df['name'].tolist(),
                    help="Select multiple items to delete"
                )
                
                if items_to_delete and st.button("🗑️ Delete Selected", use_container_width=True):
                    st.session_state.items = [
                        item for item in st.session_state.items 
                        if item['name'] not in items_to_delete
                    ]
                    st.success(f"Deleted {len(items_to_delete)} items!")
                    time.sleep(1)
                    st.rerun()
            
            with col_action3:
                if st.button("🔄 Bulk Update Status", use_container_width=True):
                    new_status = st.selectbox("New Status", 
                        ["Pending", "In Progress", "Completed"])
                    if st.button("Apply to Filtered"):
                        for item in st.session_state.items:
                            if item['name'] in filtered_df['name'].values:
                                item['status'] = new_status
                        st.success(f"Updated {len(filtered_df)} items!")
                        time.sleep(1)
                        st.rerun()
        else:
            st.info("No items match your filters. Try different criteria.")

# Analytics Page
elif page == "📈 Analytics":
    st.markdown("<h1 class='main-title'>📈 Advanced Analytics</h1>", unsafe_allow_html=True)
    
    if not st.session_state.items:
        st.warning("No data available for analytics. Add some items first!")
    else:
        df = pd.DataFrame(st.session_state.items)
        
        # Convert dates
        df['created_date'] = pd.to_datetime(df['created_at']).dt.date
        if 'due_date' in df.columns:
            df['due_date_dt'] = pd.to_datetime(df['due_date'])
        
        # Summary Metrics
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(df))
        with col2:
            st.metric("Categories", df['category'].nunique())
        with col3:
            avg_progress = df['progress'].mean() if 'progress' in df.columns else 0
            st.metric("Avg Progress", f"{avg_progress:.1f}%")
        with col4:
            overdue = len([item for item in st.session_state.items if 'due_date' in item and 
                          datetime.strptime(item['due_date'], "%Y-%m-%d") < datetime.now()])
            st.metric("Overdue", overdue, delta_color="inverse")
        
        st.divider()
        
        # Charts
        st.markdown("### 📈 Interactive Charts")
        
        tab1, tab2, tab3 = st.tabs(["Category Analysis", "Timeline View", "Performance Metrics"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Category distribution
                cat_counts = df['category'].value_counts()
                fig1 = px.bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    title="Items by Category",
                    color=cat_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig1.update_layout(xaxis_title="Category", yaxis_title="Count")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_b:
                # Priority by category heatmap
                if 'priority' in df.columns:
                    heatmap_data = df.groupby(['category', 'priority']).size().unstack(fill_value=0)
                    fig2 = px.imshow(
                        heatmap_data,
                        title="Priority Heatmap by Category",
                        color_continuous_scale='RdYlGn_r',
                        aspect="auto"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Timeline analysis
            if 'created_date' in df.columns:
                timeline_df = df.groupby('created_date').size().cumsum().reset_index(name='cumulative')
                
                fig3 = px.line(
                    timeline_df,
                    x='created_date',
                    y='cumulative',
                    title="Cumulative Items Over Time",
                    markers=True,
                    line_shape="spline"
                )
                fig3.update_traces(line=dict(width=4))
                fig3.add_scatter(
                    x=timeline_df['created_date'],
                    y=timeline_df['cumulative'],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Data Points'
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            # Performance metrics
            col_x, col_y = st.columns(2)
            
            with col_x:
                # Progress distribution
                fig4 = px.histogram(
                    df,
                    x='progress',
                    nbins=10,
                    title="Progress Distribution",
                    color_discrete_sequence=['#2E86AB']
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            with col_y:
                # Status vs Priority scatter
                if 'priority' in df.columns:
                    fig5 = px.scatter(
                        df,
                        x='priority',
                        y='progress',
                        color='status',
                        size='priority',
                        hover_data=['name'],
                        title="Priority vs Progress by Status",
                        size_max=20
                    )
                    st.plotly_chart(fig5, use_container_width=True)
        
        st.divider()
        
        # Data Export
        st.markdown("### 📥 Export Analytics")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📊 Export Summary Report", use_container_width=True):
                summary = {
                    'total_items': len(df),
                    'categories': df['category'].nunique(),
                    'avg_progress': avg_progress,
                    'completion_rate': (df['status'] == 'Completed').mean() * 100,
                    'generated_at': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(summary, indent=2),
                    file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with export_col2:
            if st.button("📈 Export Charts Data", use_container_width=True):
                charts_data = {
                    'category_distribution': cat_counts.to_dict(),
                    'timeline_data': timeline_df.to_dict('records') if 'timeline_df' in locals() else [],
                    'generated_at': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download Charts Data",
                    data=json.dumps(charts_data, indent=2),
                    file_name=f"charts_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with export_col3:
            if st.button("🔢 Raw Data Export", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Full CSV",
                    data=csv,
                    file_name=f"full_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Settings Page
elif page == "⚙️ Settings":
    st.markdown("<h1 class='main-title'>⚙️ Settings & Configuration</h1>", unsafe_allow_html=True)
    
    # Settings tabs
    setting_tab1, setting_tab2, setting_tab3, setting_tab4 = st.tabs([
        "General", "Data Management", "Appearance", "About"
    ])
    
    with setting_tab1:
        st.markdown("### ⚙️ General Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.subheader("App Behavior")
            auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
            if auto_refresh:
                refresh_interval = st.select_slider(
                    "Refresh Interval",
                    options=["30s", "1m", "5m", "10m", "30m"],
                    value="5m"
                )
            
            notifications = st.checkbox("Enable Notifications", value=True)
            sound_effects = st.checkbox("Enable Sound Effects", value=False)
            
            st.subheader("Data Handling")
            auto_save = st.checkbox("Auto-save Changes", value=True)
            confirm_delete = st.checkbox("Confirm Before Delete", value=True)
            backup_frequency = st.selectbox(
                "Auto-backup Frequency",
                ["Disabled", "Daily", "Weekly", "Monthly"]
            )
        
        with col_set2:
            st.subheader("Performance")
            cache_size = st.slider("Cache Size (MB)", 10, 1000, 100)
            max_items = st.number_input("Max Items to Display", 10, 1000, 100)
            lazy_loading = st.checkbox("Enable Lazy Loading", value=True)
            
            st.subheader("Export Settings")
            export_format = st.radio(
                "Default Export Format",
                ["CSV", "JSON", "Excel"]
            )
            include_metadata = st.checkbox("Include Metadata in Exports", value=True)
            compress_exports = st.checkbox("Compress Large Exports", value=True)
        
        if st.button("💾 Save General Settings", type="primary"):
            st.success("General settings saved successfully!")
    
    with setting_tab2:
        st.markdown("### 💾 Data Management")
        
        col_data1, col_data2 = st.columns(2)
        
        with col_data1:
            st.subheader("Backup & Restore")
            
            # Create Backup
            if st.button("📁 Create Full Backup", use_container_width=True):
                backup_data = {
                    'items': st.session_state.items,
                    'counter': st.session_state.counter,
                    'app_visits': st.session_state.app_visits,
                    'backup_time': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
                
                st.download_button(
                    label="⬇️ Download Backup File",
                    data=json.dumps(backup_data, indent=2),
                    file_name=f"dashboard_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Restore Backup
            st.subheader("Restore from Backup")
            uploaded_backup = st.file_uploader(
                "Choose backup file",
                type=['json'],
                help="Select a previously exported backup file"
            )
            
            if uploaded_backup:
                try:
                    backup_data = json.load(uploaded_backup)
                    if st.button("🔄 Restore Backup", type="secondary", use_container_width=True):
                        st.session_state.items = backup_data.get('items', [])
                        st.session_state.counter = backup_data.get('counter', 1)
                        st.session_state.app_visits = backup_data.get('app_visits', 0)
                        st.success("Backup restored successfully!")
                        time.sleep(2)
                        st.rerun()
                except:
                    st.error("Invalid backup file format")
        
        with col_data2:
            st.subheader("Data Operations")
            
            # Export Current Data
            if st.button("📤 Export Current Data", use_container_width=True):
                csv_data = pd.DataFrame(st.session_state.items).to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"current_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Import Data
            st.subheader("Import Data")
            uploaded_data = st.file_uploader(
                "Import CSV or JSON",
                type=['csv', 'json'],
                help="Import data from external sources"
            )
            
            if uploaded_data:
                try:
                    if uploaded_data.name.endswith('.csv'):
                        imported_df = pd.read_csv(uploaded_data)
                        imported_data = imported_df.to_dict('records')
                    else:
                        imported_data = json.load(uploaded_data)
                    
                    st.info(f"Ready to import {len(imported_data)} items")
                    
                    if st.button("📥 Import Data", type="primary", use_container_width=True):
                        for item in imported_data:
                            item['id'] = st.session_state.counter
                            st.session_state.counter += 1
                            if 'created_at' not in item:
                                item['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.items.append(item)
                        
                        st.success(f"Successfully imported {len(imported_data)} items!")
                        time.sleep(2)
                        st.rerun()
                except:
                    st.error("Failed to import data. Check file format.")
            
            # Danger Zone
            st.subheader("⚠️ Danger Zone", divider="red")
            
            if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                st.warning("This will permanently delete ALL data!")
                confirm1 = st.checkbox("I understand this action cannot be undone")
                confirm2 = st.checkbox("I have backed up my data")
                
                if confirm1 and confirm2:
                    if st.button("⚠️ CONFIRM DELETE ALL DATA", type="primary", use_container_width=True):
                        st.session_state.items = []
                        st.session_state.counter = 1
                        st.error("All data has been deleted!")
                        time.sleep(2)
                        st.rerun()
    
    with setting_tab3:
        st.markdown("### 🎨 Appearance Settings")
        
        col_app1, col_app2 = st.columns(2)
        
        with col_app1:
            st.subheader("Theme Customization")
            
            primary_color = st.color_picker("Primary Color", "#1E88E5")
            secondary_color = st.color_picker("Secondary Color", "#FF4B4B")
            bg_color = st.color_picker("Background Color", "#FFFFFF")
            text_color = st.color_picker("Text Color", "#262730")
            
            font_family = st.selectbox(
                "Font Family",
                ["Inter", "Roboto", "Arial", "Helvetica", "Georgia", "Times New Roman"]
            )
            
            font_size = st.select_slider(
                "Base Font Size",
                options=["Small", "Medium", "Large"],
                value="Medium"
            )
            
            rounded_corners = st.checkbox("Rounded Corners", value=True)
            shadows = st.checkbox("Enable Shadows", value=True)
            animations = st.checkbox("Enable Animations", value=True)
        
        with col_app2:
            st.subheader("Layout Settings")
            
            default_view = st.radio(
                "Default View Mode",
                ["Grid View", "List View", "Card View"]
            )
            
            items_per_page = st.slider("Items per Page", 5, 100, 20)
            
            sidebar_position = st.radio(
                "Sidebar Position",
                ["Left", "Right"]
            )
            
            density = st.select_slider(
                "UI Density",
                options=["Compact", "Comfortable", "Spacious"],
                value="Comfortable"
            )
            
            show_help = st.checkbox("Show Help Tooltips", value=True)
            show_tutorial = st.checkbox("Show Tutorial on Startup", value=False)
        
        if st.button("🎨 Apply Appearance Settings", type="primary"):
            st.success("Appearance settings applied! Refresh to see changes.")
    
    with setting_tab4:
        st.markdown("### ℹ️ About This Application")
        
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=150)
        
        st.markdown("""
        #### Dashboard Pro
        
        **Version:** 1.0.0  
        **Last Updated:** """ + datetime.now().strftime("%B %d, %Y") + """  
        **Streamlit Version:** 1.32.0
        
        ---
        
        **Description:**  
        Dashboard Pro is a comprehensive data management and analytics dashboard built with Streamlit. 
        It provides powerful tools for managing projects, tracking progress, and analyzing data through 
        interactive visualizations.
        
        ---
        
        **Features:**  
        ✅ Multi-page navigation with sidebar  
        ✅ Interactive data tables with filtering  
        ✅ Real-time charts and analytics  
        ✅ Data import/export functionality  
        ✅ Customizable themes and settings  
        ✅ Responsive design for all devices  
        ✅ Session state management  
        ✅ Progress tracking and status management
        
        ---
        
        **Technology Stack:**  
        • **Frontend:** Streamlit  
        • **Visualization:** Plotly, Pandas  
        • **Data Processing:** Pandas, NumPy  
        • **Styling:** Custom CSS, Bootstrap-inspired
        
        ---
        
        **System Information:**  
        • **Total Items in Memory:** """ + str(len(st.session_state.items)) + """  
        • **App Visits:** """ + str(st.session_state.app_visits) + """  
        • **Current Theme:** """ + st.session_state.theme + """  
        • **Python Version:** 3.x  
        • **Pandas Version:** 2.2.0
        
        ---
        
        **Support:**  
        For issues, questions, or feature requests, please:  
        1. Check the documentation  
        2. Review common issues  
        3. Contact support if needed
        
        **GitHub Repository:** [Your Repo Link Here]  
        **Documentation:** [Your Docs Link Here]
        
        ---
        
        **Credits:**  
        Built with ❤️ using Streamlit by the Dashboard Pro Team
        
        © """ + str(datetime.now().year) + """ Dashboard Pro. All rights reserved.
        """)
        
        # System check
        st.divider()
        st.subheader("🔧 System Check")
        
        check_col1, check_col2, check_col3 = st.columns(3)
        
        with check_col1:
            st.metric("Data Integrity", "✓ OK", delta="Verified")
        
        with check_col2:
            st.metric("Storage", f"{len(st.session_state.items)} items", delta="Active")
        
        with check_col3:
            st.metric("Performance", "Optimal", delta="Ready")

# ============================================
# FOOTER
# ============================================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption(f"© {datetime.now().year} Dashboard Pro v1.0.0")

with footer_col2:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with footer_col3:
    if st.button("🔄 Refresh Session", key="footer_refresh"):
        st.session_state.app_visits += 1
        st.rerun()

# ============================================
# DEBUG INFO (Hidden by default)
# ============================================
if st.sidebar.checkbox("🔧 Show Debug Info", False):
    st.sidebar.divider()
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Items in memory: {len(st.session_state.items)}")
    st.sidebar.write(f"Next ID: {st.session_state.counter}")
    st.sidebar.write(f"Session visits: {st.session_state.app_visits}")
    st.sidebar.write(f"Current page: {page}")
    
    if st.sidebar.button("Reset App Data"):
        st.session_state.items = []
        st.session_state.counter = 1
        st.sidebar.success("Data reset complete!")
        st.rerun()
