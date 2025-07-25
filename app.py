import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time
from io import BytesIO
import base64
from visualization import EnhancedDataVisualizer
from User_page import create_user, verify_user

# Page configuration
st.set_page_config(
    page_title="AI SQL Query Assistant",
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .connection-status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-status {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error-status {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info-status {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .query-result-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sample-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .sample-question:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'connection_status': 'disconnected',
        'schema_data': [],
        'connection_info': None,
        'last_connection_time': None,
        'query_history': [],
        'current_results': None,
        'current_sql': None,
        'sample_queries': [],
        'selected_viz_type': None,
        'api_status': 'unknown'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def test_api_connection():
    """Test if FastAPI backend is running and AI is initialized"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return {
                'status': 'connected',
                'ai_status': health_data.get('ai_status', 'unknown'),
                'active_connections': health_data.get('active_connections', 0),
                'cache_size': health_data.get('cache_size', 0)
            }
    except:
        pass
    return {'status': 'disconnected'}

def connect_to_database(connection_type, credentials=None, url=None):
    """Connect to PostgreSQL database via FastAPI"""
    try:
        request_data = {
            "connection_type": connection_type
        }
        
        if connection_type == "credentials":
            request_data["credentials"] = credentials
        elif connection_type == "url":
            request_data["url_connection"] = {"url": url}
        
        response = requests.post(
            f"{API_BASE_URL}/connect",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return {'status': 'error', 'message': error_detail}
            
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': 'Connection timeout. Please check your database credentials and network.'}
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'message': 'Cannot connect to API server. Please ensure FastAPI backend is running.'}
    except Exception as e:
        return {'status': 'error', 'message': f'Unexpected error: {str(e)}'}

def process_natural_query(natural_query, connection_request):
    """Process natural language query via FastAPI"""
    try:
        request_data = {
            "natural_query": natural_query,
            "connection_request": connection_request
        }
        
        response = requests.post(
            f"{API_BASE_URL}/natural-query",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return {'status': 'error', 'message': error_detail}
            
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': 'Query processing timeout. Please try a simpler question.'}
    except Exception as e:
        return {'status': 'error', 'message': f'Query processing failed: {str(e)}'}

def create_download_link(df, filename="query_results.csv"):
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def display_connection_status():
    """Display enhanced connection status in sidebar"""
    st.sidebar.subheader("Connection Status")
    
    # API Status
    api_info = test_api_connection()
    st.session_state.api_status = api_info['status']
    
    if api_info['status'] == 'connected':
        st.sidebar.markdown('<div class="connection-status success-status">✅ API Connected</div>', unsafe_allow_html=True)
        if api_info['ai_status'] == 'healthy':
            st.sidebar.markdown('<div class="connection-status success-status">AI Ready</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="connection-status error-status">❌ AI Unavailable</div>', unsafe_allow_html=True)
        
        # Show additional info
        st.sidebar.caption(f"Active connections: {api_info['active_connections']} | Cache size: {api_info['cache_size']}")
    else:
        st.sidebar.markdown('<div class="connection-status error-status">❌ API Disconnected</div>', unsafe_allow_html=True)
    
    # Database Status
    if st.session_state.connection_status == 'connected':
        st.sidebar.markdown('<div class="connection-status success-status">✅ Database Connected</div>', unsafe_allow_html=True)
        if st.session_state.connection_info:
            conn_info = st.session_state.connection_info
            st.sidebar.caption(f"Database: {conn_info['database']} | Tables: {conn_info['schema_summary']['total_tables']}")
    elif st.session_state.connection_status == 'error':
        st.sidebar.markdown('<div class="connection-status error-status">❌ Database Connection Failed</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="connection-status info-status">Database Not Connected</div>', unsafe_allow_html=True)

def display_query_history():
    """Display enhanced query history in sidebar"""
    if st.session_state.query_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Recent Queries")
        
        for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.sidebar.expander(f"Query {len(st.session_state.query_history)-i}", expanded=False):
                st.write(f"**Q:** {query['question'][:50]}...")
                st.code(query['sql'][:100] + "..." if len(query['sql']) > 100 else query['sql'], language="sql")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Rows: {query['result_count']}")
                with col2:
                    st.caption(f"Time: {query.get('execution_time', 0):.2f}s")

def display_schema_overview():
    """Display compact schema overview"""
    if st.session_state.schema_data:
        st.subheader("Database Schema")
        
        # Summary metrics
        total_tables = len(st.session_state.schema_data)
        total_columns = sum(len(table['columns']) for table in st.session_state.schema_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tables", total_tables)
        with col2:
            st.metric("Columns", total_columns)
        with col3:
            avg_columns = total_columns / total_tables if total_tables > 0 else 0
            st.metric("Avg Cols/Table", f"{avg_columns:.1f}")
        
        # Tables overview with expandable details
        for table_data in st.session_state.schema_data:
            table_name = table_data['table_name']
            columns = table_data['columns']
            estimated_rows = table_data.get('estimated_rows', 0)
            st.markdown(f"#### **{table_name}** ({len(columns)} columns, ~{estimated_rows:,} rows)")
            # Create DataFrame for better display
            df_data = []
            for col in columns:
                df_data.append({
                    'Column': col['column_name'],
                    'Type': col['data_type'],
                    'Nullable': '✅' if col['is_nullable'] == 'YES' else '❌',
                    'Default': col['column_default'] if col['column_default'] else '—'
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

def display_sample_questions():
    """Display interactive sample questions"""
    if st.session_state.sample_queries:
        st.subheader("Try These Questions")
        
        # Create columns for better layout
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.sample_queries):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.user_question = question
                    st.rerun()

def display_visualization_options(results, viz_suggestions):
    """Display enhanced visualization options"""
    st.subheader("Visualization Options")
    
    if not viz_suggestions:
        st.info("No visualization suggestions available for this data.")
        return
    
    # Display suggestions with detailed info
    st.write("**Recommended visualizations based on your data:**")
    
    cols = st.columns(min(len(viz_suggestions), 4))
    
    for i, suggestion in enumerate(viz_suggestions):
        with cols[i % 4]:
            with st.container():
                st.markdown(f"<b>{suggestion['title']}</b>", unsafe_allow_html=True)
                st.caption(suggestion['description'])
                st.caption(f"{suggestion['suitable_for']}")
                
                if st.button(f"Create {suggestion['type'].title()}", key=f"viz_{i}", use_container_width=True):
                    st.session_state.selected_viz_type = suggestion['type']
                    st.rerun()

def show_login_signup():
    st.title("🔐 User Authentication")

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if email and password:
                    success, message = verify_user(email, password)
                    if success:
                        st.success(message)
                        st.balloons()
                        # Here you can redirect to main app or set session state
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill all fields")

    with tab2:
        with st.form("signup_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            signup = st.form_submit_button("Sign Up", use_container_width=True)

            if signup:
                if not email or not password or not confirm:
                    st.warning("Please complete all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                else:
                    success, message = create_user(email, password)
                    if success:
                        st.success(message)
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)

def main():
    """Enhanced main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("AI SQL Query Assistant")
    st.markdown("Ask questions about your data in natural language - AI will handle the rest!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display connection status
    display_connection_status()
    
    # Check API connection
    if st.session_state.api_status != 'connected':
        st.error("**API Backend Unavailable**")
        st.markdown("""
        **Please start the FastAPI backend:**
        ```bash
        # Install dependencies
        pip install fastapi uvicorn psycopg2-binary streamlit pandas requests google-generativeai plotly python-dotenv
        
        # Create .env file with your Gemini API key
        echo "GEMINI_API_KEY=your_api_key_here" > .env
        
        # Start the backend
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```
        """)
        return
    
    # Sidebar for database connection
    with st.sidebar:
        st.header("Database Connection")
        
        # Connection type selector
        connection_type = st.radio(
            "Connection Method:",
            ["credentials", "url"],
            format_func=lambda x: " Credentials" if x == "credentials" else "Connection URL"
        )
        
        with st.form("connection_form"):
            if connection_type == "credentials":
                st.subheader("Database Credentials")
                host = st.text_input("Host", value="localhost", help="Database server hostname")
                port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
                database = st.text_input("Database Name*", help="PostgreSQL database name")
                username = st.text_input("Username*", help="Database username")
                password = st.text_input("Password", type="password", help="Database password")
                
                connect_button = st.form_submit_button("Connect Database", use_container_width=True)
                
                if connect_button:
                    if not database or not username:
                        st.error("❌ Database name and username are required")
                    else:
                        with st.spinner("Connecting to database..."):
                            credentials = {
                                "host": host,
                                "port": port,
                                "database": database,
                                "username": username,
                                "password": password
                            }
                            
                            result = connect_to_database("credentials", credentials=credentials)
                            handle_connection_result(result, {"connection_type": "credentials", "credentials": credentials})
            
            else:  # URL connection
                st.subheader("Connection URL")
                database_url = st.text_input(
                    "PostgreSQL URL*",
                    placeholder="postgresql://username:password@host:port/database",
                    help="Complete PostgreSQL connection string"
                )
                
                connect_button = st.form_submit_button("Connect Database", use_container_width=True)
                
                if connect_button:
                    if not database_url:
                        st.error("❌ Database URL is required")
                    else:
                        with st.spinner("Connecting to database..."):
                            result = connect_to_database("url", url=database_url)
                            handle_connection_result(result, {"connection_type": "url", "url_connection": {"url": database_url}})
        
        # Display query history
        display_query_history()
    
    # Main content area - show interface based on connection status
    if st.session_state.connection_status == 'connected':
        display_connected_interface()
    else:
        display_disconnected_interface()

def handle_connection_result(result, connection_request):
    """Handle database connection result"""
    if result['status'] == 'success':
        st.session_state.connection_status = 'connected'
        st.session_state.schema_data = result.get('schema_data', [])
        st.session_state.connection_info = result.get('connection_info', {})
        st.session_state.sample_queries = result.get('sample_queries', [])
        st.session_state.last_connection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.connection_request = connection_request
        st.success(f"✅ {result['message']}")
        st.rerun()
    else:
        st.session_state.connection_status = 'error'
        st.session_state.schema_data = []
        st.error(f"❌ {result['message']}")

def display_connected_interface():
    """Display interface when database is connected"""
    # Natural Language Query Interface
    st.header("Ask Your Data Questions")
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        if 'user_question' not in st.session_state:
            st.session_state.user_question = ""
        
        user_question = st.text_input(
            "What would you like to know?",
            value=st.session_state.user_question,
            placeholder="e.g., Show me the top 10 customers by sales, What's the total revenue this year?",
            help="Ask any question about your database in natural language",
            key="query_input"
        )
    
    with col2:
        query_button = st.button("Ask", use_container_width=True, type="primary")
    
    # Sample questions
    display_sample_questions()
    
    # Process query
    if query_button and user_question.strip():
        with st.spinner("AI is analyzing your question and generating SQL..."):
            result = process_natural_query(user_question, st.session_state.connection_request)
            
            if result['status'] == 'success':
                handle_successful_query(result, user_question)
            else:
                handle_failed_query(result, user_question)
    
    # Display current results if available
    if st.session_state.current_results:
        display_query_results()
    
    # Schema overview (collapsible)
    with st.expander("Database Schema Overview", expanded=False):
        display_schema_overview()

def handle_successful_query(result, user_question):
    """Handle successful query execution"""
    st.session_state.current_results = result
    st.session_state.current_sql = result['sql_query']
    
    # Add to history
    st.session_state.query_history.append({
        'question': user_question,
        'sql': result['sql_query'],
        'result_count': len(result['results']),
        'execution_time': result.get('execution_time', 0),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Clear the input
    st.session_state.user_question = ""
    
    # Show success message
    cache_info = " (from cache)" if result.get('from_cache') else ""
    st.success(f"✅ {result['message']}{cache_info}")

def handle_failed_query(result, user_question):
    """Handle failed query execution"""
    st.error(f"❌ {result['message']}")
    
    # Show generated SQL if available
    if result.get('sql_query'):
        with st.expander("Generated SQL (Failed)", expanded=True):
            st.code(result['sql_query'], language="sql")
    
    # Show suggestions if available
    if result.get('visualization_suggestions'):
        st.info("**Try these sample questions instead:**")
        for suggestion in result['visualization_suggestions'][:3]:
            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                st.session_state.user_question = suggestion
                st.rerun()


def display_query_results():
    """Display comprehensive query results - UPDATED VERSION"""
    result = st.session_state.current_results
    
    # Add debug button (remove this in production)
    # if st.button("Debug Data Flow"):
    #     debug_data_flow()
    
    # Results header
    st.markdown('<div class="query-result-header">', unsafe_allow_html=True)
    st.markdown("### Query Results")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show explanation
    if result.get('explanation'):
        st.info(result['explanation'])
    
    # Show generated SQL
    with st.expander("Generated SQL Query", expanded=False):
        st.code(result['sql_query'], language="sql")
    
    # Display results
    if result['results']:
        # CRITICAL: Store raw results for visualization
        raw_results = result['results']
        
        # Convert to DataFrame for table display
        df = pd.DataFrame(raw_results)
        
        # Results table
        st.subheader("Data Table")
        
        # Table controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
        with col2:
            if st.button("Download CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Save CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col3:
            show_all = st.checkbox("Show all rows")
        
        # Display table
        if show_all or len(df) <= 100:
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.dataframe(df.head(100), use_container_width=True, height=400)
            st.caption(f"Showing first 100 of {len(df):,} rows. Check 'Show all rows' to see everything.")
        
        # Visualization options
        if result.get('visualization_suggestions'):
            st.markdown("---")
            display_visualization_options(raw_results, result['visualization_suggestions'])
            
            # Create visualization if selected
            if st.session_state.selected_viz_type:
                create_selected_visualization(df, result['visualization_suggestions'])
    else:
        st.info("No data returned from your query.")


def create_selected_visualization(df, viz_suggestions):
    """Create the selected visualization - FIXED VERSION"""
    viz_type = st.session_state.selected_viz_type
    
    st.subheader(f"{viz_type.title()} Visualization")
    
    try:
        visualizer = EnhancedDataVisualizer()
        
        # Find the suggestion details
        suggestion = next((s for s in viz_suggestions if s['type'] == viz_type), None)
        
        if suggestion:
            # CRITICAL FIX: Pass the raw results, not DataFrame.to_dict('records')
            # The results should already be in the correct format from your SQL execution
            
            # Get the raw results from session state instead of converting DataFrame
            if 'current_results' in st.session_state and st.session_state.current_results:
                raw_results = st.session_state.current_results['results']
                
                
                # Pass the raw results directly - this is the key fix
                visualizer.create_visualization(
                    raw_results,  # Use raw results, not df.to_dict('records')
                    suggestion,
                    st.session_state.get('current_sql', 'Custom Visualization')
                )
            else:
                st.error("No raw results available in session state")
                # Fallback to DataFrame conversion only if raw results not available
                st.warning("Falling back to DataFrame conversion...")
                visualizer.create_visualization(
                    df.to_dict('records'),
                    suggestion,
                    "Custom Visualization (Fallback)"
                )
        else:
            # Fallback to basic visualization if no suggestion found
            st.warning("No matching suggestion found, creating basic visualization...")
            
            # For basic visualization, we can use DataFrame
            visualizer.create_basic_visualization(df, viz_type)
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("**Error details:**")
        st.write(f"- Visualization type: {viz_type}")
        st.write(f"- DataFrame shape: {df.shape}")
        st.write(f"- DataFrame columns: {df.columns.tolist()}")
        st.write(f"- DataFrame dtypes: {df.dtypes.to_dict()}")
        
        # Show the DataFrame as a table as fallback
        st.info("Showing data as table instead:")
        st.dataframe(df, use_container_width=True)
    
    # Reset selection
    if st.button("Try Another Visualization"):
        st.session_state.selected_viz_type = None
        st.rerun()



# Also, you need to modify your display_query_results function slightly:
def display_query_results():
    """Display comprehensive query results - UPDATED VERSION"""
    result = st.session_state.current_results
    
    
    # Results header
    st.markdown('<div class="query-result-header">', unsafe_allow_html=True)
    st.markdown("### Query Results")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show explanation
    if result.get('explanation'):
        st.info(result['explanation'])
    
    # Show generated SQL
    with st.expander("Generated SQL Query", expanded=False):
        st.code(result['sql_query'], language="sql")
    
    # Display results
    if result['results']:
        # CRITICAL: Store raw results for visualization
        raw_results = result['results']
        
        # Convert to DataFrame for table display
        df = pd.DataFrame(raw_results)
        
        # Results table
        st.subheader("Data Table")
        
        # Table controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
        with col2:
            if st.button("Download CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Save CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col3:
            show_all = st.checkbox("Show all rows")
        
        # Display table
        if show_all or len(df) <= 100:
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.dataframe(df.head(100), use_container_width=True, height=400)
            st.caption(f"Showing first 100 of {len(df):,} rows. Check 'Show all rows' to see everything.")
        
        # Visualization options
        if result.get('visualization_suggestions'):
            st.markdown("---")
            display_visualization_options(raw_results, result['visualization_suggestions'])
            
            # Create visualization if selected
            if st.session_state.selected_viz_type:
                create_selected_visualization(df, result['visualization_suggestions'])
    else:
        st.info("No data returned from your query.")


def display_disconnected_interface():
    """Display interface when database is not connected"""
    st.info(" **Welcome to AI SQL Query Assistant!**")
    st.markdown("""
    This app connects to your PostgreSQL database and lets you ask questions in natural language. 
    The AI will automatically generate SQL queries and visualize your results.
    """)
    
    # Quick start guide
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Connect Your Database** (sidebar):
           - Choose **Credentials** for individual connection parameters
           - Choose **Connection URL** for a complete PostgreSQL URL
        
        2. **Start Asking Questions**:
           - "Show me all customers"
           - "What's the total revenue this month?"
           - "Find the top selling products"
        
        3. **Get Instant Results**:
           - View data in tables
           - See AI-generated SQL queries
           - Get automatic visualizations
        
        ### Features
        -  **AI-Powered**: Natural language to SQL conversion
        -  **Smart Visualizations**: Automatic chart recommendations
        -  **Secure**: Read-only queries, no data modification
        -  **Fast**: Connection pooling and query caching
        -  **Export**: Download results as CSV
        """)
    
    # Connection examples
    with st.expander("Connection Examples"):
        st.markdown("""
        ### Credential Connection
        - **Host**: `localhost` or `your-server.com`
        - **Port**: `5432` (default PostgreSQL port)
        - **Database**: `your_database_name`
        - **Username**: `your_username`
        - **Password**: `your_password`
        
        ### URL Connection
        ```
        postgresql://username:password@localhost:5432/database_name
        postgres://user:pass@hostname:5432/dbname
        ```
        """)

if __name__ == "__main__":
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        show_login_signup()
    else:
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        main()
